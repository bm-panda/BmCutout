"""边缘后处理：导向滤波 + 过渡带锐化。纯 numpy，零第三方依赖。

背景：推理分辨率固定 1024，回到原图要放大 2~8 倍。放大不会产生新信息，
过渡带（半透明边缘）会同步放大 —— 这就是「边缘发虚」的直接来源。

这里做两件事，都可单点关闭：

1. ``refine_alpha``  —— 用原图做引导，把软 alpha 吸附到真实图像边缘上（仅大图）
2. ``sharpen_alpha`` —— 收窄半透明过渡带

两者都在 float 域操作（0.0~1.0），**不要在这之前量化成 uint8**。
"""

import numpy as np

# --- 开关与参数（单点收口，便于灰度与回退）---
REFINE_MIN_SIDE = 1500   # 长边小于此值不做导向滤波（放大倍数低，收益不足以抵消成本）
REFINE_EPS = 1e-4        # 导向滤波正则项，越大越平滑
REFINE_TILE = 2048       # 分块尺寸，用来压住大图的内存峰值

SHARPEN_LO = 0.20        # [lo, hi] 铺满 [0, 1]；设成 (0.0, 1.0) 即等价于关闭
SHARPEN_HI = 0.80
SHARPEN_GAMMA = 1.0


# ------------------------------------------------------------------ 基础算子
def _box_mean_1d(a, r, axis):
    """沿单轴做窗口均值，**边界 replicate**（窗口恒为 2r+1，边缘像素被重复计入）。

    先在轴上补 r 宽度的 edge padding，再对 padded 数组取定宽窗口的前缀和差分。
    这样语义与 OpenCV boxFilter / scipy uniform_filter 的默认边界一致，
    也是 guided filter 的通行做法。

    注意：这里的边界语义必须是 replicate。若改成「窗口在边缘收缩」，
    导向滤波在离边界 r 像素内的结果会偏，且分块实现与整图实现不再等价。
    """
    if r <= 0:
        return a
    n = a.shape[axis]

    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (r, r)
    ap = np.pad(a, pad_width, mode="edge")

    # ★ 累加必须走 float64。float32 前缀和在几千长度上会累积到 ~1e-4 的绝对误差，
    #   而导向滤波里 a = cov / (var + eps) 在平坦区 var≈0，误差会被 1/eps 放大到
    #   与信号同量级 —— 表现为平坦区斑块噪点，并且分块结果与整图结果不再一致。
    #   （前缀和的差分天然抵消共模误差，所以显式抬到 float64 是廉价的。）
    cs = np.cumsum(ap, axis=axis, dtype=np.float64)
    pad_shape = list(ap.shape)
    pad_shape[axis] = 1
    cs = np.concatenate([np.zeros(pad_shape, dtype=np.float64), cs], axis=axis)

    idx = np.arange(n)
    # 原图位置 i 对应 padded 的 [i, i+2r]，和 = cs[i+2r+1] - cs[i]
    out = (
        np.take(cs, idx + 2 * r + 1, axis=axis) - np.take(cs, idx, axis=axis)
    ) / (2 * r + 1)
    return out.astype(np.float32, copy=False)


def box_mean(img, r):
    """二维窗口均值，分离式（先列后行）实现。"""
    a = np.asarray(img, dtype=np.float32)
    if r <= 0:
        return a
    return _box_mean_1d(_box_mean_1d(a, r, axis=1), r, axis=0)


def guided_filter(guide, src, r, eps=REFINE_EPS):
    """He et al. 2007 导向滤波，guide / src 均为 float32 (h, w)。

    直觉：以原图（guide）的局部结构去重建软 alpha（src）。
    原图有边缘的地方 → 局部方差大 → a≈1，alpha 保持锐利；
    原图平坦的地方 → 局部方差小 → a≈0，alpha 被平滑。
    净效果是把「糊开」的 mask 边缘重新吸附到图像真实边缘上。
    """
    g = np.asarray(guide, dtype=np.float32)
    p = np.asarray(src, dtype=np.float32)

    mI = box_mean(g, r)
    mp = box_mean(p, r)
    mII = box_mean(g * g, r)
    mIp = box_mean(g * p, r)

    a = (mIp - mI * mp) / ((mII - mI * mI) + eps)
    b = mp - a * mI
    del mI, mp, mII, mIp

    q = box_mean(a, r) * g + box_mean(b, r)
    del a, b
    return q


def _guided_filter_tiled(guide, src, r, eps, tile=REFINE_TILE):
    """分块导向滤波。

    每块向外扩 **2r** 像素再计算。为什么是 2r 而不是 r：
    ``q = box_mean(a, r)·g + box_mean(b, r)``，而 ``a/b`` 本身又由 ``mI/mp/mII/mIp``
    四个半径 r 的窗口均值算出 —— 所以输出像素的等效支撑半径是 **2r**。
    只扩 r 的话，块边界处会有 2r 宽的条带与整图结果不一致（小 r 时不易察觉，
    半径一大就露馅）。

    块内每个像素的完整支撑区都在扩展区内，因此结果与整图一次性计算**精确一致**，
    只是内存峰值降到「一块」的量级。
    """
    h, w = src.shape[:2]
    out = np.empty((h, w), dtype=np.float32)
    ext = 2 * r

    for y0 in range(0, h, tile):
        y1 = min(y0 + tile, h)
        for x0 in range(0, w, tile):
            x1 = min(x0 + tile, w)
            gy0, gy1 = max(0, y0 - ext), min(h, y1 + ext)
            gx0, gx1 = max(0, x0 - ext), min(w, x1 + ext)
            sub = guided_filter(
                guide[gy0:gy1, gx0:gx1], src[gy0:gy1, gx0:gx1], r, eps
            )
            out[y0:y1, x0:x1] = sub[y0 - gy0 : y1 - gy0, x0 - gx0 : x1 - gx0]
    return out


# ------------------------------------------------------------------ 对外接口
def refine_alpha(alpha_f, src_rgb, min_side=REFINE_MIN_SIDE, tile=REFINE_TILE):
    """把软 alpha 吸附到原图真实边缘上。大图才有收益。

    alpha_f: float32 (h, w)，取值 0~1
    src_rgb: PIL RGB 图（与 alpha_f 同尺寸）
    """
    h, w = alpha_f.shape[:2]
    long_side = max(h, w)
    if long_side < min_side:
        return alpha_f

    # 半径跟着放大倍数走：推理是 1024，过渡带宽度大致按倍数等比放大。
    # 放大 8 倍 → 过渡带约 16px，取 r=16 刚好覆盖。
    scale = long_side / 1024.0
    r = int(max(4, min(64, round(scale * 2))))

    guide = np.asarray(src_rgb.convert("L"), dtype=np.float32) * (1.0 / 255.0)
    return np.clip(_guided_filter_tiled(guide, alpha_f, r, REFINE_EPS, tile), 0.0, 1.0)


def sharpen_alpha(m, lo=SHARPEN_LO, hi=SHARPEN_HI, gamma=SHARPEN_GAMMA):
    """收窄半透明过渡带：把 [lo, hi] 线性铺满 [0, 1]。

    低于 lo 一律判背景、高于 hi 一律判前景，中间的软边被拉成更陡的过渡。
    ``lo<=0 且 hi>=1 且 gamma==1`` 时是恒等变换（即关闭）。
    """
    if lo <= 0.0 and hi >= 1.0 and gamma == 1.0:
        return m
    a = np.clip((m - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if gamma != 1.0:
        a = np.power(a, gamma, out=a)
    return a
