"""不忙抠图 — 入口。

盒子启动方式：python main.py "<参数JSON路径>"
盒子把一个 JSON 文件路径塞进命令行第一个参数，里面是 environment / data / params 三段。

三条分流（判断发生在建窗口、等待、倒计时之前 —— 这是做节点的硬要求）：
    node        被别的脚本经 /api/link 调用：无人值守，只写信封，绝不弹窗/等待
    manual      右键菜单 / 卡片点击 / 快捷键
    scheduled   定时任务：无人值守，回写最小信封后退出

信封错误码（业务码，与盒子合成的 -1/-2/-3 区分开）：
    0 全部成功　1 部分成功　2 全部失败　3 模型不可用　4 输入无效
"""

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ORIG_CWD = os.getcwd()  # CLI 的相对路径按用户实际所在目录解析

# 工作目录保护：脚本所在目录不一定是运行目录（规范 §2.3）
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
try:
    os.chdir(SCRIPT_DIR)
except OSError:
    pass

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from src.box import (  # noqa: E402
    MODE_RGBA,
    OUTPUT_MODES,
    env_of,
    read_payload,
    read_version,
    resolve_params,
    target_paths,
    write_envelope,
)
from src.compose import load_image, output_path, render, save  # noqa: E402
from src.infer import MattingEngine, ModelMissingError  # noqa: E402
from src.model import resolve_model_paths  # noqa: E402
from src.ui import (  # noqa: E402
    banner,
    batch_warning,
    error,
    log,
    pause_before_exit,
    progress,
)

NODE = "node"
MANUAL = "manual"
SCHEDULED = "scheduled"


class BusinessError(RuntimeError):
    """业务失败（输入不对等），不是程序崩溃。"""


# ------------------------------------------------------------------ 核心流程
def _prepare(payload, log_fn=log):
    """解析参数 → 定位模型 → 建引擎。模型问题在这里一次性抛出，不再逐张重试。"""
    env = env_of(payload)
    params = resolve_params(payload)
    files = target_paths(payload)

    if not files:
        raise BusinessError("没有待处理的图片（data.target_paths 为空）")

    model_path, config_path, source = resolve_model_paths(
        env.get("api_base"), SCRIPT_DIR, warn=log_fn
    )
    if not model_path:
        raise ModelMissingError(
            "未找到 BiRefNet 模型。请在盒子中重新安装或更新本脚本以自动下载模型"
            f"（查找结果：{source}）"
        )

    engine = MattingEngine(model_path, config_path, log=log_fn)
    return params, files, engine, source


def _run_files(files, params, engine, log_fn=log):
    """逐张处理。单张失败只记录，不中断批量。"""
    mode = params["output_mode"]
    bg_color = params["bg_color"]
    out_dir = params["output_dir"]
    total = len(files)
    results, failed = [], []

    for idx, src in enumerate(files, 1):
        name = os.path.basename(src)
        started = time.perf_counter()
        try:
            if not os.path.isfile(src):
                raise FileNotFoundError("文件不存在")
            img, _ = load_image(src)
            alpha = engine.predict(img)
            dst = output_path(src, out_dir, mode)
            save(render(img, alpha, mode, bg_color), dst)
            results.append(dst)
            progress(idx, total, name, dst, time.perf_counter() - started)
        except Exception as exc:
            failed.append(src)
            error(f"[{idx}/{total}] {name} 处理失败：{type(exc).__name__}: {exc}")

    return results, failed


def _summary(results, failed, total):
    if results and not failed:
        return f"抠图完成，成功 {total}/{total} 张"
    if results:
        return f"抠图完成，成功 {len(results)}/{total} 张，失败 {len(failed)} 张"
    return f"抠图失败，{total} 张全部未成功"


def _code_for(results, failed):
    if results and not failed:
        return 0
    if results:
        return 1
    return 2


# ------------------------------------------------------------------ 节点分支
def run_node(payload):
    """被 /api/link 调用：无界面、无等待，结果只走信封。"""
    env = env_of(payload)
    out = env.get("output_json")
    if not out:
        return 1  # 契约缺 output_json，盒子会按 -3 处理

    results, failed = [], []
    try:
        params, files, engine, source = _prepare(payload, log)
        log(f"模型来源：{source}")
        log(engine.describe())
        results, failed = _run_files(files, params, engine)
        body = {
            "code": _code_for(results, failed),
            "msg": _summary(results, failed, len(files)),
            "results": results,
            "failed": failed,
        }
    except BusinessError as exc:
        body = {"code": 4, "msg": str(exc), "results": [], "failed": []}
    except ModelMissingError as exc:
        body = {"code": 3, "msg": str(exc), "results": [], "failed": []}
    except Exception as exc:
        body = {
            "code": 2,
            "msg": f"{type(exc).__name__}: {exc}",
            "results": [],
            "failed": [],
        }

    # 业务失败也必须 exit 0 —— 否则盒子会另合成错误信封，把 msg 盖掉
    return 0 if write_envelope(out, body) else 1


# ------------------------------------------------------------------ 有界面分支
def run_interactive(payload, allow_pause=True):
    """manual / scheduled：控制台展示结果。"""
    env = env_of(payload)
    invoke_mode = env.get("invoke_mode", MANUAL)

    params = resolve_params(payload)
    files = target_paths(payload)
    banner(
        read_version(SCRIPT_DIR),
        params["output_mode"],
        params["bg_color"],
        params["output_dir"],
        len(files),
    )
    batch_warning(len(files))

    results, failed, code, msg = [], [], 4, ""
    try:
        params, files, engine, source = _prepare(payload, log)
        log(f"模型来源：{source}")
        log(engine.describe())
        log()
        results, failed = _run_files(files, params, engine)
        code = _code_for(results, failed)
        msg = _summary(results, failed, len(files))
        log()
        log(msg)
    except BusinessError as exc:
        code, msg = 4, str(exc)
        error(msg)
    except ModelMissingError as exc:
        code, msg = 3, str(exc)
        error(msg)
    except Exception as exc:
        code, msg = 2, f"{type(exc).__name__}: {exc}"
        error(msg)

    # 定时任务是无人值守的：回写一个最小信封，便于盒子记录结果
    out = env.get("output_json")
    if out and invoke_mode == SCHEDULED:
        write_envelope(
            out, {"code": code, "msg": msg, "results": results, "failed": failed}
        )

    # 进程退出码沿用生态惯例：只要有一张成功就算 0
    exit_code = 0 if results else 1

    if allow_pause and invoke_mode != SCHEDULED:
        pause_before_exit(5)
    return exit_code


# ------------------------------------------------------------------ CLI（调试）
USAGE = f"""BM-Cutout — BiRefNet 本地抠图（不忙脚本盒子脚本）

用法:
  盒子模式: python main.py <参数JSON路径>
  CLI 调试: python main.py <图片...> [-m {"|".join(OUTPUT_MODES)}] [-o 输出目录] [--no-pause]
"""


def run_cli(argv):
    parser = argparse.ArgumentParser(
        prog="main.py", description="BiRefNet 本地抠图（CLI 调试模式）"
    )
    parser.add_argument("images", nargs="+", help="图片文件路径")
    parser.add_argument(
        "-m",
        "--mode",
        choices=OUTPUT_MODES,
        default=MODE_RGBA,
        help="输出形式",
    )
    parser.add_argument("--bg-color", default="#FFFFFF", help="自定义底色的十六进制值")
    parser.add_argument("-o", "--out", default="", help="输出目录（默认原图同目录）")
    parser.add_argument("--no-pause", action="store_true", help="结束后不等待")
    opts = parser.parse_args(argv)

    # chdir 之后相对路径会按脚本目录解析，这里还原成用户实际所在目录
    images = [
        p if os.path.isabs(p) else os.path.normpath(os.path.join(_ORIG_CWD, p))
        for p in opts.images
    ]
    payload = {
        "environment": {"invoke_mode": MANUAL},
        "data": {"target_paths": images},
        "params": {
            "output_mode": opts.mode,
            "bg_color": opts.bg_color,
            "output_dir": opts.out,
        },
    }
    return run_interactive(payload, allow_pause=not opts.no_pause)


def _entry():
    args = sys.argv[1:]

    if args and args[0].lower().endswith(".json"):
        payload = read_payload(args[0])
        if not payload:
            log("✗ 无法读取参数文件（文件不存在、不是 JSON，或不是 JSON 对象）")
            return 1
        env = env_of(payload)
        if env.get("invoke_mode") == NODE:
            return run_node(payload)  # 必须在建窗口/等待之前分流
        return run_interactive(payload)

    if args:
        return run_cli(args)

    print(USAGE, file=sys.stderr)
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(_entry())
