from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


MODEL_FILENAME = "UVR-MDX-NET-Inst_HQ_5.onnx"
MODEL_SAMPLE_RATE = 44100
MODEL_DIM_F = 2560
MODEL_DIM_T = 8
MODEL_N_FFT = 5120
MODEL_HOP_LENGTH = 1024
MODEL_COMPENSATION = 1.01
DEFAULT_SEGMENT_SECONDS = 15.0
DEFAULT_MARGIN_SECONDS = 1.0
DEFAULT_DEVICE = "auto"


_SESSION_CACHE: dict[tuple[str, str], Any] = {}


def _require_torch():
    import torch
    import torch.nn.functional as F

    return torch, F


def _require_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "This node requires onnxruntime. Install either onnxruntime or onnxruntime-gpu in your ComfyUI Python environment."
        ) from exc
    return ort


def _validate_audio_input(audio: dict[str, Any]) -> tuple[Any, int]:
    if not isinstance(audio, dict):
        raise RuntimeError("Expected a ComfyUI AUDIO dict.")
    if "waveform" not in audio or "sample_rate" not in audio:
        raise RuntimeError("AUDIO input must contain 'waveform' and 'sample_rate'.")

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])

    if getattr(waveform, "ndim", None) != 3:
        raise RuntimeError("Expected waveform shape [B, C, T].")
    if sample_rate <= 0:
        raise RuntimeError("sample_rate must be greater than zero.")

    return waveform, sample_rate


def _prepare_stereo_waveform(audio: dict[str, Any], batch_index: int = 0) -> tuple[np.ndarray, int]:
    torch, _ = _require_torch()
    waveform, sample_rate = _validate_audio_input(audio)

    batch_size = int(waveform.shape[0])
    if batch_index < 0 or batch_index >= batch_size:
        raise RuntimeError(f"batch_index {batch_index} is out of range for batch size {batch_size}.")

    sample = waveform[batch_index].detach().to(torch.float32).cpu()
    channels = int(sample.shape[0])

    if channels == 1:
        sample = sample.repeat(2, 1)
    elif channels >= 2:
        sample = sample[:2]
    else:
        raise RuntimeError("Audio input must have at least one channel.")

    return np.ascontiguousarray(sample.numpy()), sample_rate


def _resample_audio(waveform: np.ndarray, src_sr: int, dst_sr: int, target_length: int | None = None) -> np.ndarray:
    if src_sr == dst_sr and (target_length is None or waveform.shape[-1] == target_length):
        return waveform.astype(np.float32, copy=False)

    torch, F = _require_torch()
    tensor = torch.from_numpy(np.ascontiguousarray(waveform)).to(torch.float32).unsqueeze(0)
    if target_length is None:
        target_length = max(1, int(round(waveform.shape[-1] * dst_sr / src_sr)))
    resampled = F.interpolate(tensor, size=int(target_length), mode="linear", align_corners=False)
    return np.ascontiguousarray(resampled.squeeze(0).cpu().numpy())


def _to_comfy_audio(waveform: np.ndarray, sample_rate: int) -> dict[str, Any]:
    torch, _ = _require_torch()
    tensor = torch.from_numpy(np.ascontiguousarray(waveform)).to(torch.float32).unsqueeze(0)
    return {"waveform": tensor, "sample_rate": int(sample_rate)}


def _resolve_model_path() -> Path:
    node_dir = Path(__file__).resolve().parent
    candidate = (node_dir / "models" / MODEL_FILENAME).resolve()
    if candidate.is_file():
        return candidate

    raise RuntimeError(
        f"Bundled model not found: {candidate}\n"
        f"Place {MODEL_FILENAME} in the node's models folder before starting ComfyUI."
    )


def _select_provider(device: str) -> tuple[str, list[str]]:
    ort = _require_onnxruntime()
    available = set(ort.get_available_providers())
    requested = device.strip().lower()

    provider_priority = {
        "auto": ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider"],
        "directml": ["DmlExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }
    if requested not in provider_priority:
        raise RuntimeError(f"Unsupported device '{device}'. Use auto, cuda, directml, or cpu.")

    for provider in provider_priority[requested]:
        if provider in available:
            return provider, [provider]

    available_text = ", ".join(sorted(available)) or "none"
    raise RuntimeError(f"No compatible onnxruntime provider found for device='{device}'. Available providers: {available_text}")


def _get_session(model_path: Path, device: str):
    ort = _require_onnxruntime()
    provider_name, providers = _select_provider(device)
    cache_key = (str(model_path), provider_name)
    session = _SESSION_CACHE.get(cache_key)
    if session is None:
        session = ort.InferenceSession(str(model_path), providers=providers)
        _SESSION_CACHE[cache_key] = session
    return session, provider_name


class _MDXModel:
    def __init__(self) -> None:
        torch, _ = _require_torch()
        self.dim_c = 4
        self.dim_f = MODEL_DIM_F
        self.dim_t = 2**MODEL_DIM_T
        self.n_fft = MODEL_N_FFT
        self.hop = MODEL_HOP_LENGTH
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = self.hop * (self.dim_t - 1)
        self.trim = self.n_fft // 2
        self.gen_size = self.chunk_size - 2 * self.trim
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.freq_pad = torch.zeros([1, self.dim_c, self.n_bins - self.dim_f, self.dim_t], dtype=torch.float32)

    def stft(self, x):
        torch, _ = _require_torch()
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, self.dim_c, self.n_bins, self.dim_t])
        return x[:, :, : self.dim_f]

    def istft(self, x):
        torch, _ = _require_torch()
        freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1])
        x = torch.cat([x, freq_pad], dim=-2)
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.permute([0, 2, 3, 1]).contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
        )
        return x.reshape([-1, 2, self.chunk_size])


class _MDXSeparator:
    def __init__(self, model_path: Path, device: str, denoise: bool) -> None:
        self.model_path = model_path
        self.device = device
        self.denoise = denoise
        self.model = _MDXModel()
        self.session, self.provider_name = _get_session(model_path, device)

    def separate(
        self,
        mix: np.ndarray,
        segment_seconds: float,
        margin_seconds: float,
        compensation: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if mix.ndim != 2:
            raise RuntimeError("Expected audio array with shape [C, T].")

        segment_samples = int(round(max(0.0, segment_seconds) * MODEL_SAMPLE_RATE))
        margin_samples = int(round(max(0.0, margin_seconds) * MODEL_SAMPLE_RATE))

        segmented_mix = self._segment_mix(mix, segment_samples=segment_samples, margin_samples=margin_samples)
        instrumental = self._demix_base(segmented_mix, margin_samples=margin_samples)
        if compensation > 0:
            instrumental = instrumental * np.float32(compensation)

        vocals = mix - instrumental
        return instrumental.astype(np.float32, copy=False), vocals.astype(np.float32, copy=False)

    def _segment_mix(self, mix: np.ndarray, segment_samples: int, margin_samples: int) -> dict[int, np.ndarray]:
        samples = int(mix.shape[-1])
        if samples == 0:
            raise RuntimeError("Input audio is empty.")

        if segment_samples <= 0 or samples < segment_samples:
            segment_samples = samples

        if margin_samples > segment_samples:
            margin_samples = segment_samples

        segmented: dict[int, np.ndarray] = {}
        counter = -1
        for skip in range(0, samples, segment_samples):
            counter += 1
            start_margin = 0 if counter == 0 else margin_samples
            end = min(skip + segment_samples + margin_samples, samples)
            start = max(0, skip - start_margin)
            segmented[skip] = np.ascontiguousarray(mix[:, start:end])
            if end >= samples:
                break
        return segmented

    def _run_session(self, spek: np.ndarray) -> np.ndarray:
        if self.denoise:
            negative = self.session.run(None, {"input": -spek})[0]
            positive = self.session.run(None, {"input": spek})[0]
            return (-negative * 0.5) + (positive * 0.5)
        return self.session.run(None, {"input": spek})[0]

    def _demix_base(self, mixes: dict[int, np.ndarray], margin_samples: int) -> np.ndarray:
        torch, _ = _require_torch()

        outputs: list[np.ndarray] = []
        keys = list(mixes.keys())

        for index, key in enumerate(keys):
            cmix = mixes[key]
            n_sample = int(cmix.shape[1])
            trim = self.model.trim
            gen_size = self.model.gen_size
            pad = (gen_size - (n_sample % gen_size)) % gen_size
            mix_p = np.concatenate(
                [
                    np.zeros((2, trim), dtype=np.float32),
                    cmix.astype(np.float32, copy=False),
                    np.zeros((2, pad), dtype=np.float32),
                    np.zeros((2, trim), dtype=np.float32),
                ],
                axis=1,
            )

            windows: list[np.ndarray] = []
            cursor = 0
            while cursor < n_sample + pad:
                windows.append(np.array(mix_p[:, cursor : cursor + self.model.chunk_size], copy=False))
                cursor += gen_size

            mix_waves = torch.tensor(np.stack(windows), dtype=torch.float32)
            spek = self.model.stft(mix_waves).cpu().numpy().astype(np.float32, copy=False)
            pred = self._run_session(spek)
            tar_waves = self.model.istft(torch.tensor(pred, dtype=torch.float32))

            tar_signal = tar_waves[:, :, trim:-trim].permute(1, 0, 2).reshape(2, -1).cpu().numpy()
            if pad > 0:
                tar_signal = tar_signal[:, :-pad]

            start = 0 if index == 0 else margin_samples
            end = None if index == len(keys) - 1 else -margin_samples
            if margin_samples == 0:
                end = None
            outputs.append(np.ascontiguousarray(tar_signal[:, start:end]))

        return np.concatenate(outputs, axis=-1).astype(np.float32, copy=False)


class UVRMDXInstHQ5Node:
    CATEGORY = "audio/uvr"
    FUNCTION = "separate"
    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("instrumental_audio", "vocals_audio", "info")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "audio": ("AUDIO",),
                "device": (["auto", "cuda", "directml", "cpu"], {"default": DEFAULT_DEVICE}),
                "segment_seconds": ("FLOAT", {"default": DEFAULT_SEGMENT_SECONDS, "min": 0.0, "max": 120.0, "step": 0.5}),
                "margin_seconds": ("FLOAT", {"default": DEFAULT_MARGIN_SECONDS, "min": 0.0, "max": 10.0, "step": 0.1}),
                "denoise": ("BOOLEAN", {"default": True}),
                "compensation": ("FLOAT", {"default": MODEL_COMPENSATION, "min": 0.0, "max": 4.0, "step": 0.001}),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            }
        }

    def separate(
        self,
        audio: dict[str, Any],
        device: str,
        segment_seconds: float,
        margin_seconds: float,
        denoise: bool,
        compensation: float,
        batch_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        source_waveform, source_sr = _prepare_stereo_waveform(audio, batch_index=batch_index)
        resolved_model_path = _resolve_model_path()

        working_waveform = _resample_audio(source_waveform, source_sr, MODEL_SAMPLE_RATE)
        separator = _MDXSeparator(resolved_model_path, device=device, denoise=denoise)
        instrumental, vocals = separator.separate(
            working_waveform,
            segment_seconds=segment_seconds,
            margin_seconds=margin_seconds,
            compensation=compensation,
        )

        target_length = source_waveform.shape[-1]
        if source_sr != MODEL_SAMPLE_RATE:
            instrumental = _resample_audio(instrumental, MODEL_SAMPLE_RATE, source_sr, target_length=target_length)
            vocals = _resample_audio(vocals, MODEL_SAMPLE_RATE, source_sr, target_length=target_length)

        info = (
            f"model={resolved_model_path.name} | provider={separator.provider_name} | "
            f"sample_rate={source_sr} | denoise={denoise} | compensation={compensation:.3f}"
        )
        return _to_comfy_audio(instrumental, source_sr), _to_comfy_audio(vocals, source_sr), info


NODE_CLASS_MAPPINGS = {
    "UVRMDXInstHQ5": UVRMDXInstHQ5Node,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "UVRMDXInstHQ5": "UVR MDX Inst HQ5 Separator",
}
