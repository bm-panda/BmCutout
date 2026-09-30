"""合成与落盘。

统一走 PIL 而不是 OpenCV：Windows 下 cv2.imwrite 用的是 C 的 fopen，
对中文/非 ASCII 路径会**静默失败**（返回 False 或写出空文件）；
PIL 走 Python 的 open，配合盒子注入的 PYTHONUTF8=1 稳。

本模块还负责**边缘去污染**：半透明边缘像素的颜色里混着原背景色，
换背景后会漏出来（白边/灰边光晕）。详见 estimate_foreground。
"""

import os

import numpy as np
from PIL import Image, ImageOps

from .box import MODE_COLOR, MODE_MASK, MODE_RGBA, MODE_WHITE
from .edges import box_mean

# 各输出模式的文件名后缀（统一 .png，全部无损）
SUFFIX = {
    MODE_RGBA: "_cutout",
    MODE_WHITE: "_cutout_white",
    MODE_COLOR: "_cutout_bg",
    MODE_MASK: "_cutout_mask",
}

# --- 去污染开关与参数（单点收口，便于灰度与回退）---
DECONTAMINATE = True            # 关掉即回到 v1.0 的「直接配 alpha」行为
DECONTAM_TAU = 0.95             # alpha >= tau 视为「可信核心」，其颜色是纯前景色
DECONTAM_RADII = (4, 16, 64)    # 外推半径序列，逐级放大直到覆盖过渡带
BAND_LO = 0.02                  # alpha 低于此值算纯背景，不参与修复


def load_image(path):
    """读图并纠正 EXIF 方向（手机竖拍不纠正会歪）。返回 (RGB 图, (W,H))。"""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
    return im, im.size


def parse_hex_color(text, default=(255, 255, 255)):
    """'#F5F5F5' / 'F5F5F5' / '#fff' → (r,g,b)；解析失败用 default。"""
    s = (text or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        return default
    try:
        return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return default


def estimate_foreground(rgb, a, tau=DECONTAM_TAU, radii=DECONTAM_RADII):
    """估计干净前景色，消掉原背景色的污染。

    半透明边缘像素满足 ``C = F·α + B_old·(1-α)`` —— 它的 RGB 里混着**原背景色**
    ``B_old``（手机拍多半是亮背景，所以表现为白边/灰边）。换到新背景后 ``B_old``
    就漏出来了。

    做法：``α >= tau`` 的像素几乎不透明，其颜色可信（就是纯前景 ``F``）。对过渡带
    像素，取其周围半径 ``r`` 内可信核心色的均值；``r`` 从 4 逐级放大到 64，直到覆盖
    整个过渡带。核心像素保持原样，纯背景不参与、也不被修改。

    参数：
        rgb: float32 (h, w, 3)，取值 0~255
        a:   float32 (h, w)，取值 0~1
    返回：float32 (h, w, 3)，去污染后的前景色
    """
    core = a >= tau
    band = (a > BAND_LO) & (~core)
    if not core.any() or not band.any():
        return rgb  # 没有可信核心、或压根没有过渡带 —— 无需处理

    mask = core.astype(np.float32)
    # 初值取原图：核心像素原样保留，纯背景像素也不该被改动
    # （它们 alpha=0 不参与合成，但把 RGB 抹成黑色会在下游产生边缘杂色）
    out = np.array(rgb, dtype=np.float32, copy=True)
    filled = core.copy()

    for r in radii:
        remaining = band & (~filled)
        if not remaining.any():
            break
        den = box_mean(mask, r)
        valid = remaining & (den > 0.0)
        if not valid.any():
            continue
        for c in range(3):
            num = box_mean(rgb[..., c] * mask, r)
            np.copyto(out[..., c], num / np.maximum(den, 1e-6), where=valid)
        filled |= valid

    # 过渡带比最大外推半径还宽的部分：保持原色（out 初值即原色），不再处理
    return out


def render(src_rgb, alpha, mode, bg_color="#FFFFFF", decontaminate=DECONTAMINATE):
    """按输出模式合成。返回 PIL 图（MODE_MASK 返回 'L'，其余返回 RGB/RGBA）。"""
    if mode == MODE_MASK:
        return alpha  # 蒙版是纯 alpha，去污染不得触碰

    rgb = src_rgb.convert("RGB")
    if decontaminate:
        a = np.asarray(alpha, dtype=np.float32) * (1.0 / 255.0)
        f = estimate_foreground(np.asarray(rgb, dtype=np.float32), a)
        rgba = np.dstack(
            [np.clip(f, 0.0, 255.0).astype(np.uint8), np.asarray(alpha, dtype=np.uint8)]
        )
        fg = Image.fromarray(rgba, "RGBA")
    else:
        fg = rgb.convert("RGBA")
        fg.putalpha(alpha)

    if mode == MODE_RGBA:
        return fg

    bg_rgb = (255, 255, 255) if mode == MODE_WHITE else parse_hex_color(bg_color)
    bg = Image.new("RGBA", src_rgb.size, bg_rgb + (255,))
    return Image.alpha_composite(bg, fg).convert("RGB")


def output_path(src_path, out_dir, mode):
    """决定输出文件路径：默认原图同目录；out_dir 非空则放那儿。"""
    stem = os.path.splitext(os.path.basename(src_path))[0]
    folder = out_dir.strip() if out_dir and out_dir.strip() else os.path.dirname(src_path)
    return os.path.join(folder, f"{stem}{SUFFIX.get(mode, '_cutout')}.png")


def ensure_dir(path):
    """确保输出目录存在（定时任务场景 out_dir 可能是用户手填的新目录）。"""
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)


def save(img, path):
    """落盘。显式指定 format，避免扩展名歧义；中文路径安全。"""
    ensure_dir(path)
    img.save(path, format="PNG")
    return path
