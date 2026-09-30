"""BiRefNet_lite ONNX 推理内核：会话构造 / 输出自省 / 预处理 / 后处理。

模型规格（onnx-community/BiRefNet_lite-ONNX，2026-09-30 实测自省确认）：
    输入   float32 [1, 3, 1024, 1024]，RGB，节点名 input_image
           ⚠ 尺寸以 **ONNX 图** 为准，preprocessor_config.json 可能过期/写错
    归一化 来自 preprocessor_config.json：
           do_rescale=true、rescale_factor=1/255、do_normalize=true、
           mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]（ImageNet 归一化）
           ⚠ 与 IS-Net 完全不同（那边是 do_rescale=false、mean=128、std=256）
    输出   float32 [1, 1, 1024, 1024]，节点名 output_image
           **是 logits 不是概率图** —— 实测纯灰图 [-13.7, -9.2]、白底蓝圆 [-18.9, +16.3]。
           需补 sigmoid；但这一点是**逐仓库属性**，见 model.REPO_SPECS。
    输入尺寸 **静态 1024×1024** —— 实测喂 512/1536 直接 INVALID_ARGUMENT，
           所以不能靠加大输入分辨率来提精度（要更高精度得走局部二次推理）。

后处理链路（全在 float 域，最后才量化）：
    logits → sigmoid → BILINEAR 上采样回原图 → 大图导向滤波 → 过渡带锐化 → uint8
"""

import os

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError as e:  # pragma: no cover - 依赖缺失时给可读信息
    ort = None
    _ORT_IMPORT_ERROR = e
else:
    _ORT_IMPORT_ERROR = None

from .edges import refine_alpha, sharpen_alpha
from .model import (  # noqa: F401  (re-export，方便调用方少导一层)
    REPO_ID,
    get_repo_spec,
    load_preprocess_cfg,
)

# HF 的 PILImageResampling 与 PIL 的常量同编号
_PIL_RESAMPLE = {
    0: Image.NEAREST,
    1: Image.LANCZOS,
    2: Image.BILINEAR,
    3: Image.BICUBIC,
    4: Image.BOX,
    5: Image.HAMMING,
}

# 输出名优先级：BiRefNet 的导出叫 output_image；其余为常见别名兜底
_PREFER_OUTPUT_NAMES = (
    "output_image",
    "mask",
    "output_mask",
    "output",
    "logits",
    "out",
    "d1",
    "d0",
)

MAX_THREADS = 8

# 实测（6 核 CPU、onnxruntime CPU EP）：intra=4 约 14s，intra=8 约 20s。
# 超线程/超订阅会让这个内存带宽敏感的模型明显变慢，所以线程数不超过 8。
#
# ⚠ 内存：BiRefNet_lite@1024 推理峰值实测约 6.3 GB（IS-Net 只有 735 MB）。
# 原因是解码器在 1024×1024 上跑 256 通道的 fp32 特征图，单张就 1 GB，同时活几张。
# 换 SessionOptions（mem_pattern / opt_level / arena）都只能挪动几个百分点，是架构性的。
# 因此加载前先看一眼可用物理内存，不够就明确告警 —— 宁可提示，也不要跑到一半被 OOM 杀掉。
MIN_FREE_MB = 7000

# 是否补 sigmoid **不是全局常量，而是逐仓库属性**，见 model.REPO_SPECS。
# 不要改成 "auto"：它靠「值是否超出 [0,1]」判断，而极端平坦的图上 logits 可能整体
# 落在 [-1e-3, 1+1e-3] 内 → 被误判成概率图 → 全图 ≈0.5，抠图彻底坏掉。
# 反过来，对已经含 sigmoid 的导出再补一次，输出会全塌到 0.5 —— 两个方向都会静默出错。

# alpha 上采样插值。**不要用 LANCZOS** —— 它会在边缘产生过冲振铃，
# 叠加 4~8 倍放大后就是一圈灰/黑光晕。BILINEAR 单调、无过冲，且保留软过渡。
ALPHA_UPSAMPLE = Image.BILINEAR


class ModelMissingError(RuntimeError):
    """模型不可用（盒子没下过 / 文件缺失）。属业务失败，不是崩溃。"""


def resample_to_pil(value):
    """config 里的整数 resample → PIL 常量；未知值退化为 BILINEAR。"""
    return _PIL_RESAMPLE.get(int(value), Image.BILINEAR)


def _cpu_threads(images_in_flight=1):
    cores = os.cpu_count() or 4
    return max(1, min(cores // max(1, images_in_flight), MAX_THREADS))


def available_mb():
    """系统可用物理内存（MB）。取不到返回 None（不阻塞主流程）。"""
    try:
        import ctypes
        from ctypes import wintypes

        class _MEMSTAT(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = _MEMSTAT()
        st.dwLength = ctypes.sizeof(st)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return st.ullAvailPhys / 1048576.0
    except Exception:  # 非 Windows 或 API 不可用 —— 静默跳过
        return None


# ------------------------------------------------------------------ 预处理 / 后处理
def preprocess(img, cfg):
    """PIL.Image(RGB) → NCHW float32 (1,3,H,W)。

    复刻 HF 的处理顺序：resize →（可选 rescale）→ normalize。
    BiRefNet 是 do_rescale=true + ImageNet 归一化；
    IS-Net 是 do_rescale=false + mean=128/std=256。两者都由 cfg 驱动，不硬编码。
    """
    if cfg["do_resize"]:
        w, h = cfg["size_wh"]
        im = img.convert("RGB").resize((w, h), resample_to_pil(cfg["resample"]))
    else:
        im = img.convert("RGB")

    x = np.asarray(im, dtype=np.float32)
    if cfg["do_rescale"]:
        x = x * cfg.get("rescale_factor", 1.0 / 255.0)
    if cfg["do_normalize"]:
        x = (x - np.asarray(cfg["mean"], dtype=np.float32)) / np.asarray(
            cfg["std"], dtype=np.float32
        )

    x = np.transpose(x, (2, 0, 1))[None, ...]
    # ORT 对非连续数组会先拷一份，这里直接给连续的
    return np.ascontiguousarray(x, dtype=np.float32)


def alpha_from_logits(out, sigmoid="on"):
    """网络输出 → float32 alpha (h, w)，取值 0~1。**不做量化**。

    sigmoid: "on" 强制补 / "off" 不补 / "auto" 按值域判断。
    """
    m = np.asarray(out, dtype=np.float32)
    m = np.squeeze(m)
    if m.ndim == 3:  # (C,H,W) 或 (H,W,C)，取通道维
        m = m[0] if m.shape[0] == 1 else m[..., 0]
    if m.ndim != 2:
        raise ValueError(f"无法理解的网络输出形状: {np.asarray(out).shape}")

    if sigmoid == "on":
        need = True
    elif sigmoid == "off":
        need = False
    else:
        lo, hi = float(m.min()), float(m.max())
        need = lo < -1e-3 or hi > 1.0 + 1e-3
    if need:
        m = 1.0 / (1.0 + np.exp(-m))

    # 不做 min-max 拉伸 —— rembg 的 (p-min)/(max-min) 在整图平坦时除爆，
    # 且会破坏标定概率。
    return np.clip(m, 0.0, 1.0)


def upsample_alpha_float(m, size_wh, resample=ALPHA_UPSAMPLE):
    """在 float 域把 alpha 放大到原图尺寸。

    先量化成 uint8 再放大会引入台阶，叠加 LANCZOS 的振铃 —— 那是「边缘发虚 +
    光晕」的直接来源。'F' 模式让 PIL 直接对 float 数据做插值，无量化损失。
    """
    if not size_wh or tuple(m.shape[::-1]) == tuple(size_wh):
        return np.asarray(m, dtype=np.float32)
    return np.asarray(
        Image.fromarray(np.asarray(m, dtype=np.float32), mode="F").resize(
            tuple(size_wh), resample
        ),
        dtype=np.float32,
    )


def to_alpha(out, size_wh, sigmoid="on"):
    """网络输出 → 原图尺寸的 alpha（PIL 'L' 图）。

    只做 sigmoid + 上采样，不含导向滤波与锐化 —— 那两步在 MattingEngine.predict
    里按图尺寸条件启用（这里是可独立调用的纯函数）。
    """
    m = upsample_alpha_float(alpha_from_logits(out, sigmoid), size_wh)
    return Image.fromarray((m * 255.0 + 0.5).astype(np.uint8), mode="L")


# ------------------------------------------------------------------ 输出自省
def pick_output_name(sess, log=None):
    """从会话里挑出真正的 alpha 输出名。

    策略：唯一输出 → 直接用；否则按名字优先级 → 唯一的单通道输出 → 空间维最大者。
    """
    outs = sess.get_outputs()
    names = [o.name for o in outs]
    if log:
        log(f"onnx 输出（{len(outs)} 个）: {names}")

    if len(outs) == 1:
        return outs[0].name

    low = {o.name.lower(): o.name for o in outs}
    for cand in _PREFER_OUTPUT_NAMES:
        if cand in low:
            if log:
                log(f"按名字选中输出: {low[cand]}")
            return low[cand]

    def channels(o):
        shape = o.shape
        return shape[1] if len(shape) == 4 and isinstance(shape[1], int) else None

    single = [o for o in outs if channels(o) == 1]
    if len(single) == 1:
        if log:
            log(f"按唯一单通道输出选中: {single[0].name}")
        return single[0].name

    def area(o):
        shape = o.shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            return shape[2] * shape[3]
        return -1

    best = max(outs, key=area)
    if log:
        log(f"⚠ 无法按名字/通道判定，改用空间维最大的输出: {best.name}")
    return best.name


# ------------------------------------------------------------------ 引擎
class MattingEngine:
    """一次加载、多张复用。ONNX 会话不是线程安全的，串行使用。"""

    def __init__(self, model_path, config_path=None, log=None):
        if ort is None:
            raise ModelMissingError(f"onnxruntime 未安装：{_ORT_IMPORT_ERROR}")
        if not model_path or not os.path.isfile(model_path):
            raise ModelMissingError(f"模型文件不存在：{model_path}")

        self._log = log
        self.cfg = load_preprocess_cfg(config_path, warn=log)
        spec = get_repo_spec(REPO_ID, warn=log)
        self.sigmoid = spec["sigmoid"]
        if log:
            log(f"输出量纲: {spec['note']} → sigmoid={self.sigmoid}")

        # 内存闸门：不够就先说清楚，别等到被 OOM 杀掉再报个看不懂的错
        free = available_mb()
        if free is not None and free < MIN_FREE_MB and log:
            log(
                f"⚠ 当前可用内存约 {free:.0f} MB，本模型推理峰值约需 6300 MB。"
                f"建议先关掉浏览器等占内存的程序，否则可能很慢甚至失败。"
            )

        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # 只留 warning/error
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = _cpu_threads(images_in_flight=1)
        opts.inter_op_num_threads = 1  # 串行算子链，inter 无用
        # 关掉 CPU 内存池。实测（6 核，BiRefNet_lite@1024，峰值 MB / 单张秒）：
        #     arena=关 intra=4 -> 6286 / 14.1     arena=开 intra=4 -> 7763 / 18.6
        #     arena=关 intra=8 -> 6283 / 19.8     arena=开 intra=8 -> 8934 / 23.3
        # 关掉既省内存又不慢，两个指标都赢。（IS-Net 上省得更多，结论一致。）
        opts.enable_cpu_mem_arena = False

        self.sess = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

        inp = self.sess.get_inputs()[0]
        self.in_name, self.in_type = inp.name, inp.type
        self.out_name = pick_output_name(self.sess, log=log)

        # ★ 输入尺寸以 **ONNX 图** 为权威。preprocessor_config.json 里的 size 是
        #   HF 处理器的提示，可能过期或写错（实测 Phoenix-ActuallyFree 的 512 导出
        #   就写成了 1024）。信 config 会直接喂错形状。
        shape = list(inp.shape)
        if (
            len(shape) == 4
            and isinstance(shape[2], int) and isinstance(shape[3], int)
            and shape[2] > 0 and shape[3] > 0
        ):
            graph_wh = (shape[3], shape[2])
            if graph_wh != tuple(self.cfg["size_wh"]):
                if log:
                    log(f"⚠ 模型实际输入 {graph_wh[0]}×{graph_wh[1]}，配置声明 "
                        f"{self.cfg['size_wh'][0]}×{self.cfg['size_wh'][1]} —— 以模型为准")
                self.cfg["size_wh"] = graph_wh
        elif log:
            log("⚠ 模型输入含动态维度，按配置尺寸处理")

        self._logged = False

    def describe(self):
        sz = f"{self.cfg['size_wh'][0]}×{self.cfg['size_wh'][1]}"
        return (
            f"输入 {self.in_name}:{self.in_type} | 输出 {self.out_name} | "
            f"尺寸 {sz} | 归一化 mean={self.cfg['mean']} std={self.cfg['std']} "
            f"rescale={self.cfg['do_rescale']}"
        )

    def _feeds(self, img):
        x = preprocess(img, self.cfg)
        if "uint8" in self.in_type:
            # 量化导出的常见形态：整数量纲输入，不做 normalize
            w, h = self.cfg["size_wh"]
            arr = np.asarray(
                img.convert("RGB").resize((w, h), resample_to_pil(self.cfg["resample"])),
                dtype=np.uint8,
            )
            x = np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, ...])
        elif "float16" in self.in_type:
            x = x.astype(np.float16)
        return {self.in_name: x}

    def predict(self, img):
        """PIL RGB 图 → 原图尺寸的 alpha（PIL 'L' 图）。"""
        src = img.convert("RGB")
        out = self.sess.run([self.out_name], self._feeds(src))[0]

        a = alpha_from_logits(out, sigmoid=self.sigmoid)
        a = upsample_alpha_float(a, src.size)
        a = refine_alpha(a, src)   # 大图：把软边吸附到原图真实边缘（小图内部直接返回）
        a = sharpen_alpha(a)       # 收窄半透明过渡带
        return Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), mode="L")
