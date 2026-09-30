# 不忙抠图
> 基于 BiRefNet_lite 高精度二分类图像分割模型的本地 AI 抠图脚本：纯 CPU 推理，全程离线运行；支持通过“不忙脚本盒子”一键配置环境与模型；支持右键选中图片进行批量抠图和定时抠图


![不忙抠图演示](./imgs/demo.gif "不忙抠图演示")

> 演示图展示「原图 → 蒙版 → 透明底」的效果

---

## 核心特性

- **全离线**：模型和推理都在本机，图片不出你的电脑
- **一键安装**：通过不忙脚本盒子自动安装、不需要自己装 Python、CUDA 或任何 AI 框架，全程自动处理
- **干净边缘**：发丝、绒毛级别的边缘做了两步专门处理
- **四种输出**：透明底 PNG、纯白底、自定义底色（电商/证件照常用）、单独蒙版
- **批量处理**：一次可选中多张图片，逐张处理，单张失败不影响其他
- **三种入口**：右键菜单、盒子卡片、定时任务，也能被其他脚本当作节点调用
- **支持格式**：PNG、JPG/JPEG、BMP、WEBP、TIF/TIFF



## 安装方法
> 本脚本图形化窗口渲染、环境配置、模型下载均依赖[不忙脚本盒子](https://www.bm-box.cn)、仅建议使用盒子安装

1. 打开 **不忙脚本盒子**
2. 在脚本社区搜索**不忙抠图**
3. 点击安装、等待安装完成 
4. 装好后在脚本列表里就能看到「不忙抠图」

> 首次安装耗时主要在下载模型，装好后再次运行无需等待。

## 使用方式

### 右键菜单抠图（最常用）

1. 在资源管理器里选中一张或多张图片
2. 右键 → **不忙抠图**
3. 盒子弹出的启动窗口里选好「输出」形式，点执行
4. 结果默认输出到图片所在目录，文件名加后缀

### 卡片点击运行

在盒子里点「不忙抠图」卡片，同样弹出启动窗口。在「源图片」处选文件即可。

### 定时任务

脚本支持定时任务。因为是无人值守运行，**建议把「输出目录」填成绝对路径**（例如 `D:\抠图结果`）——定时任务没有右键上下文，留空会输出到源图所在目录。

### 输出文件名

| 输出形式 | 文件名后缀 |
| --- | --- |
| 透明PNG | `_cutout.png` |
| 白底图 | `_cutout_white.png` |
| 自定义底色 | `_cutout_bg.png` |
| 仅蒙版 | `_cutout_mask.png` |

例如 `照片.jpg` → `照片_cutout.png`。

## 使用提示

- **⚠ 内存**：单张推理**峰值约 6.3 GB**（不是笔误）。模型解码器要在 1024×1024 上跑 256 通道的
  浮点特征图，单张就 1 GB。**建议在 16 GB 内存的机器上使用**；8 GB 机器请先关闭浏览器等
  占内存的程序，否则会非常慢甚至失败。脚本启动时会检查可用内存并在不足时给出提示。
- **速度**：6 核 CPU 上单张约 **14 秒**（含模型加载，首次会多几秒）。图片大小基本不影响
  耗时（统一在 1024×1024 上推理），所以批量 10 张大约两分半。
- **超大图**：模型工作在 1024×1024，6000px 级别的图会先缩小再放大蒙版，发丝级别的细节会有
  损失。如果对超大图边缘要求很高，建议先裁剪到需要的主体区域再抠。
- **半透明物体**：玻璃、纱质、烟这类**主体内部**就该半透明的东西，会被过渡带锐化压成较硬的
  边。需要保留这种通透感时，把 `src/edges.py` 里的 `SHARPEN_LO/HI` 调回 `0.0/1.0` 即可关闭。
- **EXIF**：手机竖拍的照片会自动纠正方向，不会出现歪图。

## 作为节点被其他脚本调用（开发者）

本脚本声明了 `[bmscriptsbox.node]`，可以被其他盒子脚本通过 `POST /api/link` 同步调用。

```python
import json, sys, requests

payload = json.load(open(sys.argv[1], encoding="utf-8"))
env = payload["environment"]

r = requests.post(f"{env['api_base']}/api/link", json={
    "script_id": "523b6d94-7955-48fa-9fb9-b0efb17370c5",
    "data": {"target_paths": ["D:/图片/a.png", "D:/图片/b.jpg"]},
    "params": {"output_mode": "透明PNG", "bg_color": "#FFFFFF", "output_dir": ""},
}, timeout=600)

body = r.json()
if not body.get("success"):
    raise RuntimeError(body.get("message"))

result = body["result"]          # 被调脚本写回的信封
if result.get("code") != 0:
    raise RuntimeError(result.get("msg"))

print(result["results"])         # 输出文件路径列表
print(result["failed"])          # 失败的源文件列表
```

**信封字段**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `code` | int | 见下表 |
| `msg` | str | 人类可读的结果说明 |
| `results` | list | 成功输出的图片路径 |
| `failed` | list | 处理失败的源文件路径 |

**业务错误码**（与盒子合成的 `-1/-2/-3` 区分开）

| code | 含义 |
| --- | --- |
| `0` | 全部成功 |
| `1` | 部分成功（有失败项） |
| `2` | 全部失败 |
| `3` | 模型不可用（未下载 / 文件缺失） |
| `4` | 输入无效（没有待处理图片） |

## 环境要求

- Windows 10 / 11
- 不忙脚本盒子
- 内存：建议 16 GB（单张推理峰值约 6.3 GB）
- Python 3.12（盒子自动部署，无需手动安装）
  - 必须 ≥ 3.12：`onnxruntime` 自 1.24 起已不再提供 CPython 3.10 的 wheel，
    而盒子会按 `language_version` 锁定运行时次版本，因此 3.12 是本脚本的可用档位
- 磁盘：模型约 214 MB + 运行环境约 200 MB

## 关于模型

模型为 **BiRefNet_lite**（Bilateral Reference for High-Resolution Dichotomous Image
Segmentation 的轻量版）通用抠图模型，通过盒子从 huggingface 下载
[`onnx-community/BiRefNet_lite-ONNX`](https://huggingface.co/onnx-community/BiRefNet_lite-ONNX)。

- 模型**不包含在脚本包内**，由盒子在安装时下载到本地并全局共享
- 原模型出处：[ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)，**MIT 许可**
- 输入 **1024×1024** 静态尺寸；输出为 logits，需补 sigmoid（见 `src/model.py` 的 `REPO_SPECS`）
- 导出差异提示：不同仓库的 ONNX 导出**输出量纲可能不同**（logits vs 概率图），
  且 `preprocessor_config.json` 里的 `size` 可能过期。本脚本以 **ONNX 图的实际
  输入尺寸为权威**，并按仓库登记后处理方式，换模型时务必同步 `REPO_SPECS`。


