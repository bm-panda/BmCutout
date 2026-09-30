"""盒子契约层：读参数 JSON、参数兜底、写信封。

JSON 三段契约（盒子 0.3.0）：
  environment — 盒子信息，脚本只读：api_base / output_json / invoke_mode / script_dir / encoding
  data        — 业务数据，键名 = TOML [[inputs]].name，值恒为字符串数组
  params      — 运行参数，键名 = TOML [[params]].name

本模块同时是「输出模式」的单一真源 —— rc.toml 里 output_mode 的 choices 必须与
OUTPUT_MODES 完全一致（tools/check_rc.py 会做集合比对，防止两处手抄漂移）。
"""

import json
import os

# ------------------------------------------------------------------ 单一真源
MODE_RGBA = "透明PNG"
MODE_WHITE = "白底图"
MODE_COLOR = "自定义底色"
MODE_MASK = "仅蒙版"

OUTPUT_MODES = [MODE_RGBA, MODE_WHITE, MODE_COLOR, MODE_MASK]

# 参数默认值（TOML 里也声明了一份，盒子会注入；这里兜底防缺）
DEFAULTS = {
    "output_mode": MODE_RGBA,
    "bg_color": "#FFFFFF",
    "output_dir": "",
}

INPUT_NAME = "target_paths"


def read_payload(path):
    """读 argv[1] 的参数 JSON。读不到/格式不对返回空 dict（不抛）。"""
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def env_of(payload):
    env = payload.get("environment")
    return env if isinstance(env, dict) else {}


def resolve_params(payload):
    """取出运行参数；未声明/被 when 隐藏的键用 DEFAULTS 兜底。

    被 when 隐藏的参数不会注入脚本，所以这里必须 .get 兜底。
    """
    raw = payload.get("params")
    raw = raw if isinstance(raw, dict) else {}
    params = dict(DEFAULTS)
    for key in DEFAULTS:
        if key in raw and raw[key] is not None:
            params[key] = raw[key]

    if params["output_mode"] not in OUTPUT_MODES:
        params["output_mode"] = DEFAULTS["output_mode"]

    return params


def target_paths(payload):
    """取主输入（TOML [[inputs]].name）的路径数组。"""
    data = payload.get("data")
    data = data if isinstance(data, dict) else {}
    paths = data.get(INPUT_NAME) or []
    if isinstance(paths, str):
        paths = [paths]
    return [p for p in paths if isinstance(p, str) and p.strip()]


def write_envelope(path, body):
    """把结果信封写进 environment.output_json。

    盒子只认 code(int) + msg(str)，业务键在顶层平铺。
    返回 True 表示写成功（此时进程应 exit 0，成败由 code 表达）；
    返回 False 表示写不进去（目录不存在/没权限），进程应 exit 1 让盒子按 -3 报错。
    """
    if not isinstance(body.get("code"), int):
        body["code"] = -1
    if not isinstance(body.get("msg"), str):
        body["msg"] = str(body.get("msg") or "")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, indent=2)
        return True
    except OSError:
        return False


def read_version(script_dir, fallback="1.0.0"):
    """从 pyproject.toml 读版本号（与 rc.toml/README 三处保持一致的单一入口）。"""
    try:
        with open(os.path.join(script_dir, "pyproject.toml"), encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("version"):
                    return line.split('"')[1]
    except (OSError, IndexError):
        pass
    return fallback
