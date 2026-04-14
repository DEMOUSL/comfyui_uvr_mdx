# ComfyUI UVR MDX

这是一个用于 ComfyUI 的人声 / 伴奏分离自定义节点，内置使用 `UVR-MDX-NET-Inst_HQ_5.onnx` 模型。

[English README](./README.md)

## 功能特点

- 可直接在 ComfyUI 中使用，不依赖 UVR 桌面程序
- 支持 `auto`、`cuda`、`directml`、`cpu` 四种执行方式
- 输出 `instrumental_audio` 和 `vocals_audio`
- 推理前自动重采样到 `44.1 kHz`，输出时再还原到原始采样率
- 单声道输入会自动复制成双声道后再分离

## 节点信息

- 节点名称：`UVR MDX Inst HQ5 Separator`
- 分类：`audio/uvr`

## 输入参数

- `audio`：ComfyUI 的 `AUDIO`
- `device`：`auto` / `cuda` / `directml` / `cpu`
- `segment_seconds`：长音频分块长度，默认 `15`
- `margin_seconds`：分块重叠长度，默认 `1`
- `denoise`：是否启用 UVR 常见的 denoise 推理方式
- `compensation`：输出补偿系数，默认 `1.01`
- `batch_index`：当输入是批量 `AUDIO` 时读取第几个

## 输出内容

- `instrumental_audio`
- `vocals_audio`
- `info`

## 依赖

ComfyUI 通常已经自带 `torch`，这个节点额外需要：

- `numpy`
- 一种 ONNX Runtime：
- `onnxruntime`：CPU 环境
- `onnxruntime-gpu`：NVIDIA CUDA 环境
- `onnxruntime-directml`：Windows DirectML 环境

## 安装方法

1. 把仓库复制或克隆到 `ComfyUI/custom_nodes/` 下。
2. 确认模型文件位于 `models/UVR-MDX-NET-Inst_HQ_5.onnx`。
3. 在 ComfyUI 使用的 Python 环境中安装依赖：

```bash
pip install -r requirements.txt
```

如果你想使用 GPU，请把默认的 ONNX Runtime 替换成适合自己环境的版本。

## 仓库结构

```text
comfyui_uvr_mdx/
|- __init__.py
|- nodes.py
|- README.md
|- README.zh-CN.md
|- requirements.txt
|- LICENSE
`- models/
   `- UVR-MDX-NET-Inst_HQ_5.onnx
```

## 模型来源说明

本节点使用的 `UVR-MDX-NET-Inst_HQ_5.onnx` 模型来自 Ultimate Vocal Remover（UVR）项目的官方分发包。

参考来源：

- 官网：[ultimatevocalremover.com](https://ultimatevocalremover.com/)
- 官方仓库：[Anjok07/ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)

根据 UVR 官方仓库 README，项目分发包中提供的模型原则上由 UVR 核心开发者训练，除非另有说明；对于使用这些模型的第三方开发者，UVR 要求提供对 UVR 及其开发者的署名。

UVR 在仓库中对 MDX-Net 原始 AI 代码的署名为：

- Kuielab
- Woosung Choi

如需重新分发仓库内附带的模型权重，请先自行确认上游模型的许可与再分发条款。

## 说明

- 节点会从本地 `models/` 目录读取模型，不依赖任何绝对路径。
- 当前仓库直接带了模型文件。虽然还没超过 GitHub 单文件 `100 MB` 限制，但如果后续模型变大，更适合放到 GitHub Releases 或单独下载。
- `LICENSE` 仅明确覆盖本仓库源码，除非另外说明。

## 常见问题

- `Bundled model not found`：把 `UVR-MDX-NET-Inst_HQ_5.onnx` 放进 `models/`
- `This node requires onnxruntime`：在 ComfyUI 的 Python 环境安装兼容的 ONNX Runtime
- `cuda` 或 `directml` 报 provider 错误：改成 `auto`，或安装正确的 Runtime 版本

## 许可证

仓库源码使用 MIT 协议，见 [LICENSE](./LICENSE)。
