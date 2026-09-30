"""控制台展示：banner / 进度 / 倒计时退出。

日志一律走 stderr。理由：盒子在 auto/never 终端模式下会接管 stdout，
而且 node 模式下 stdout 是黑洞 —— 统一 stderr 既能被盒子捕获，也方便盒子外调试。
"""

import os
import sys


def log(msg=""):
    print(msg, file=sys.stderr, flush=True)


def banner(version, mode, bg_color, out_dir, count):
    log("=" * 58)
    log(f"  BM-Cutout v{version} — BiRefNet 本地抠图")
    detail = f"  输出: {mode}"
    if mode == "自定义底色":
        detail += f"（{bg_color}）"
    detail += f" | 待处理: {count} 张"
    log(detail)
    log(f"  输出目录: {out_dir.strip() or '原图同目录'}")
    log("=" * 58)
    log()


def batch_warning(count):
    if count <= 1:
        return
    log("批量抠图会持续占用 CPU。为保证电脑操作流畅、避免任务意外中断，")
    log("建议提交后先关闭其他非必需的大型应用。")
    log()


def progress(i, total, name, dst="", seconds=0.0):
    tail = f" -> {dst}（{seconds:.2f}s）" if dst else ""
    log(f"[{i}/{total}] {name}{tail}")


def warn(msg):
    log(f"  ⚠ {msg}")


def error(msg):
    log(f"  ✗ {msg}")


def pause_before_exit(timeout=5):
    """有 TUI 的脚本按规范：暂停 → 倒计时 → 自动退出。

    不能用后台线程去阻塞 input()：线程是 daemon，解释器退出时它仍持有 stdin 锁，
    会抛 "_enter_buffered_busy: could not acquire lock" 致命错误。
    这里在主线程用 msvcrt 轮询，按键或超时即返回。
    """
    log()
    log(f"按任意键退出（{timeout} 秒后自动退出）...")
    try:
        if os.name == "nt":
            import msvcrt
            import time

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if msvcrt.kbhit():
                    msvcrt.getwch()
                    return
                time.sleep(0.05)
        else:
            import select

            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if ready:
                sys.stdin.readline()
    except Exception:
        # 非交互环境（管道/重定向）直接跳过等待
        pass
