"""模型路径解析 + 预处理配置读取。

模型由盒子在安装时统一下载到
    {盒子目录}/BmPackages/BmModels/{repo_id}/
（保留远程子目录结构），运行时用
    GET {api_base}/api/model/path?repo_id=...
换本地文件夹路径。接口只返回文件夹，具体文件名要自己拼。

盒子外（开发调试）的兜底顺序：
    1) 盒子接口
    2) 环境变量 BM_CUTOUT_MODEL_DIR —— 指向与远程同构的目录
       （含 preprocessor_config.json 与 onnx/model.onnx）
    3) {script_dir}/models/{repo_id}
    都拿不到 → 返回 None，由调用方明确报错（本脚本不做脚本内下载）。
"""

import json
import os
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

REPO_ID = "onnx-community/BiRefNet_lite-ONNX"
MODEL_REL = "onnx/model.onnx"
CONFIG_REL = "preprocessor_config.json"

ENV_MODEL_DIR = "BM_CUTOUT_MODEL_DIR"

_TIMEOUT = 5

# 每个导出的**输出量纲**不一样，必须逐仓库登记，实测确定、禁止猜测。
# 判据：喂一张纯灰图看输出的值域。
#     logits   -> 值域跑到 [-18, +16] 这种量级，必须补 sigmoid
#     概率图   -> 值域落在 [0, 1]，**再补一次 sigmoid 会全塌成 0.5，抠图彻底坏掉**
# 换个仓库忘了登记 → 静默出废图，所以下面 get_repo_spec 会显式告警。
REPO_SPECS = {
    # 实测 2026-09-30：纯灰 [-13.69,-9.25]、随机噪声 [-16.00,-9.02]、白底蓝圆 [-18.87,+16.27]
    "onnx-community/BiRefNet_lite-ONNX": {"sigmoid": "on", "note": "输出 logits"},
    # 实测 2026-09-30：纯灰 [0,0.0001]、随机噪声 [0,0.0002]、真实人像 [0,1]
    "Phoenix-ActuallyFree/BiRefNet_lite_512-ONNX": {
        "sigmoid": "off", "note": "输出已含 sigmoid 的概率图"
    },
}

_DEFAULT_SPEC = {"sigmoid": "on", "note": "未登记，按 logits 处理"}


def get_repo_spec(repo_id, warn=None):
    """取该仓库导出的后处理约定。未登记时用默认值并告警。"""
    spec = REPO_SPECS.get(repo_id)
    if spec is None:
        if warn:
            warn(f"⚠ 模型仓库 {repo_id} 未在 REPO_SPECS 登记，按默认（{_DEFAULT_SPEC['note']}）处理。"
                 f"若输出异常，请先用纯灰图确认其输出是 logits 还是概率图。")
        return dict(_DEFAULT_SPEC)
    return dict(spec)


def _query_box_dir(api_base, repo_id):
    """问盒子要模型文件夹。404 = 盒子没下过这个模型 → None（不抛）。"""
    url = f"{api_base}/api/model/path?repo_id={quote(repo_id)}"
    try:
        with urlopen(url, timeout=_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        # 404 是「未提供」，属于预期分支；其他 HTTP 错误按拿不到处理
        if e.code == 404:
            return None
        raise
    if not body.get("success"):
        return None
    path = (body.get("data") or {}).get("path")
    return path if path and os.path.isdir(path) else None


def _dev_candidates(script_dir):
    cands = []
    env_dir = os.environ.get(ENV_MODEL_DIR)
    if env_dir:
        cands.append(env_dir)
    if script_dir:
        cands.append(os.path.join(script_dir, "models", *REPO_ID.split("/")))
    return cands


def resolve_model_dir(api_base, script_dir, warn=None):
    """定位模型目录。返回 (目录 或 None, 来源说明)。"""
    if api_base:
        try:
            got = _query_box_dir(api_base, REPO_ID)
            if got:
                return got, "盒子下发"
        except Exception as e:  # 网络/协议异常都不该让脚本直接崩
            if warn:
                warn(f"查询盒子模型路径失败（{REPO_ID}）：{type(e).__name__}: {e}")

    for cand in _dev_candidates(script_dir):
        if os.path.isdir(cand):
            return cand, f"本地兜底（{os.path.basename(os.path.dirname(cand))}）"

    return None, "未找到"


def resolve_model_paths(api_base, script_dir, warn=None):
    """解析出模型文件与预处理配置的绝对路径。

    返回 (model_path, config_path)。两者都可能为 None —— 只有 model_path 是硬要求，
    config_path 缺失时由 load_preprocess_cfg 用常量兜底。
    """
    model_dir, source = resolve_model_dir(api_base, script_dir, warn=warn)
    if not model_dir:
        return None, None, source

    model_path = os.path.join(model_dir, *MODEL_REL.split("/"))
    if not os.path.isfile(model_path):
        return None, None, f"{source}（缺 {MODEL_REL}）"

    config_path = os.path.join(model_dir, *CONFIG_REL.split("/"))
    return model_path, (config_path if os.path.isfile(config_path) else None), source


# --------------------------------------------------------------------- 预处理配置
# 兜底常量 = onnx-community/BiRefNet_lite-ONNX 的 preprocessor_config.json 同值。
#
# ⚠ 换模型时必须同步改这里。它与 IS-Net 完全不同：
#     IS-Net   —— do_rescale=false，mean=[128,128,128]，std=[256,256,256]
#     BiRefNet —— do_rescale=true， mean=[0.485,0.456,0.406]，std=[0.229,0.224,0.225]
# 一旦 config 读不到又没同步常量，就会**静默按旧模型归一化** ——
# 输出整体偏色/失败，而且不报错，极难排查。所以下面 config 缺失时会显式告警。
_FB_MEAN = (0.485, 0.456, 0.406)
_FB_STD = (0.229, 0.224, 0.225)
_FB_SIZE_WH = (1024, 1024)
_FB_RESAMPLE = 2  # HF PILImageResampling 与 PIL 同编号：2 = BILINEAR
_FB_RESCALE = 1.0 / 255.0


def load_preprocess_cfg(config_path, warn=None):
    """读 preprocessor_config.json；读不到或字段缺失则用常量兜底。

    读文件而不是硬编码，是为了将来换导出/换仓库时不用改代码。
    """
    cfg = {
        "mean": list(_FB_MEAN),
        "std": list(_FB_STD),
        "size_wh": _FB_SIZE_WH,
        "resample": _FB_RESAMPLE,
        "do_resize": True,
        "do_rescale": True,
        "do_normalize": True,
        "rescale_factor": _FB_RESCALE,
    }
    if not config_path:
        # BiRefNet 的 config 随模型一起下发（见 TOML 的 models.files）。
        # 正常情况拿得到；拿不到说明下载不完整或换了仓库，值得显式提一句。
        if warn:
            warn(f"未找到 {CONFIG_REL}，改用内置默认值（BiRefNet 规格）")
        return cfg

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        if warn:
            warn(f"读取预处理配置失败，改用内置默认值：{type(e).__name__}: {e}")
        return cfg

    if raw.get("image_mean"):
        cfg["mean"] = [float(v) for v in raw["image_mean"]]
    if raw.get("image_std"):
        cfg["std"] = [float(v) for v in raw["image_std"]]
    size = raw.get("size") or {}
    if size.get("width") and size.get("height"):
        cfg["size_wh"] = (int(size["width"]), int(size["height"]))
    if raw.get("resample") is not None:
        cfg["resample"] = int(raw["resample"])
    if raw.get("rescale_factor") is not None:
        cfg["rescale_factor"] = float(raw["rescale_factor"])
    for key in ("do_resize", "do_rescale", "do_normalize"):
        if raw.get(key) is not None:
            cfg[key] = bool(raw[key])
    return cfg
