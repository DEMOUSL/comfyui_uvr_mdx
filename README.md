# ComfyUI UVR MDX

ComfyUI custom node for separating instrumental and vocal stems with the bundled `UVR-MDX-NET-Inst_HQ_5.onnx` model.

[Simplified Chinese README](./README.zh-CN.md)

## Features

- Ready to use inside ComfyUI without depending on the UVR desktop application
- Supports `auto`, `cuda`, `directml`, and `cpu` execution providers
- Splits input audio into `instrumental_audio` and `vocals_audio`
- Automatically resamples audio to `44.1 kHz` for inference and restores the original sample rate on output
- Handles mono input by duplicating it to stereo before separation

## Node

- Node name: `UVR MDX Inst HQ5 Separator`
- Category: `audio/uvr`

## Inputs

- `audio`: ComfyUI `AUDIO`
- `device`: `auto` / `cuda` / `directml` / `cpu`
- `segment_seconds`: chunk size for long audio, default `15`
- `margin_seconds`: overlap between chunks, default `1`
- `denoise`: enables the common UVR denoise inference trick
- `compensation`: output gain compensation, default `1.01`
- `batch_index`: which item to read from a batched `AUDIO` input

## Outputs

- `instrumental_audio`
- `vocals_audio`
- `info`

## Requirements

ComfyUI usually already provides `torch`. This node additionally needs:

- `numpy`
- One ONNX Runtime package:
  - `onnxruntime` for CPU
  - `onnxruntime-gpu` for NVIDIA CUDA
  - `onnxruntime-directml` for DirectML on Windows

## Installation

1. Copy or clone this repository into your `ComfyUI/custom_nodes/` folder.
2. Make sure the model file exists at `models/UVR-MDX-NET-Inst_HQ_5.onnx`.
3. Install dependencies in the same Python environment used by ComfyUI:

```bash
pip install -r requirements.txt
```

If you want GPU acceleration, replace the default ONNX Runtime package with the variant that matches your system.

## Repository Layout

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

## Model Attribution

This node uses the bundled `UVR-MDX-NET-Inst_HQ_5.onnx` model from the Ultimate Vocal Remover (UVR) distribution.

Sources:

- Official website: [ultimatevocalremover.com](https://ultimatevocalremover.com/)
- Official repository: [Anjok07/ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)

According to the UVR repository, the models provided in the package were trained by the UVR core developers unless otherwise noted, and third-party developers using UVR models should provide credit to UVR and its developers.

MDX-Net architecture credits listed by UVR:

- Kuielab
- Woosung Choi

Please verify upstream model redistribution terms before republishing bundled weights.

## Notes

- The node loads the model from its local `models/` directory and does not rely on absolute paths.
- The bundled model file is large enough that you may prefer GitHub Releases or another distribution method in the future, but it is still below GitHub's 100 MB file limit.
- The `LICENSE` file applies to the source code in this repository unless otherwise noted.
- Before redistributing the bundled model, verify the upstream model's license and redistribution terms.

## Troubleshooting

- `Bundled model not found`: place `UVR-MDX-NET-Inst_HQ_5.onnx` inside `models/`
- `This node requires onnxruntime`: install a compatible ONNX Runtime package in the ComfyUI Python environment
- Provider errors on `cuda` or `directml`: switch `device` to `auto` or install the correct runtime package for your hardware

## License

MIT for the repository source code. See [LICENSE](./LICENSE).
