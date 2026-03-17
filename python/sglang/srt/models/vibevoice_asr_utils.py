import copy
import json
import logging
import math
import os
import threading
import warnings
import wave
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from subprocess import run
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pybase64
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, BatchEncoding
from transformers.activations import ACT2FN
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import TensorType
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio input into text output in JSON format."
)
COMMON_AUDIO_EXTS = [
    ".mp3",
    ".m4a",
    ".mp4",
    ".wav",
    ".aac",
    ".ogg",
    ".mov",
    ".opus",
    ".flac",
    ".wma",
    ".webm",
]
AUDIO_SAMPLE_RATE = 24000
HAS_FFMPEG_UTILS = True
logger = logging.getLogger(__name__)


def _get_ffmpeg_max_concurrency() -> int:
    value = os.getenv("VIBEVOICE_FFMPEG_MAX_CONCURRENCY", "")
    try:
        return int(value) if value.strip() else 0
    except ValueError:
        return 0


_FFMPEG_MAX_CONCURRENCY = _get_ffmpeg_max_concurrency()
_FFMPEG_SEM = (
    threading.Semaphore(_FFMPEG_MAX_CONCURRENCY)
    if _FFMPEG_MAX_CONCURRENCY > 0
    else None
)


def _run_ffmpeg(cmd: list[str], *, stdin_bytes: bytes | None = None):
    if _FFMPEG_SEM is None:
        return run(cmd, capture_output=True, check=True, input=stdin_bytes)
    with _FFMPEG_SEM:
        return run(cmd, capture_output=True, check=True, input=stdin_bytes)


def _resample_audio_fast(
    audio: np.ndarray, original_sr: int, target_sr: int
) -> np.ndarray:
    if original_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)

    from scipy.signal import resample

    num_samples = int(len(audio) * float(target_sr) / original_sr)
    return np.asarray(resample(audio, num_samples), dtype=np.float32)


def load_audio_bytes_use_fast_wav(
    data: bytes, *, target_sr: int = AUDIO_SAMPLE_RATE
) -> tuple[np.ndarray, int]:
    with wave.open(BytesIO(data), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE":
            raise ValueError("Compressed WAV is not supported by fast loader")

        original_sr = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        pcm = wav_file.readframes(frame_count)

    if sample_width == 1:
        audio = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if original_sr != target_sr:
        audio = _resample_audio_fast(audio, original_sr, target_sr)
        original_sr = target_sr

    return np.asarray(audio, dtype=np.float32), original_sr


def load_audio_data_url_fast(
    audio_url: str, *, target_sr: int = AUDIO_SAMPLE_RATE
) -> tuple[np.ndarray, int]:
    if not audio_url.startswith("data:"):
        raise ValueError("Expected data URL")

    header, encoded = audio_url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Expected base64 data URL")

    mime = header[5:].split(";", 1)[0].lower()
    if mime not in {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}:
        raise ValueError(f"Unsupported fast-path data URL mime type: {mime}")

    audio_bytes = pybase64.b64decode(encoded, validate=True)
    return load_audio_bytes_use_fast_wav(audio_bytes, target_sr=target_sr)


def load_audio_use_ffmpeg(
    file: str, resample: bool = False, target_sr: int = AUDIO_SAMPLE_RATE
) -> tuple[np.ndarray, int]:
    if not resample:
        cmd_probe = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file,
        ]
        original_sr = int(
            run(cmd_probe, capture_output=True, check=True).stdout.decode().strip()
        )
    else:
        original_sr = None

    sr_to_use = target_sr if resample else original_sr
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr_to_use),
        "-",
    ]
    out = _run_ffmpeg(cmd).stdout
    audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    return audio, sr_to_use


def load_audio_bytes_use_ffmpeg(
    data: bytes, *, resample: bool = True, target_sr: int = AUDIO_SAMPLE_RATE
) -> tuple[np.ndarray, int]:
    if not resample:
        raise ValueError("load_audio_bytes_use_ffmpeg requires resample=True")
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-threads",
        "0",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(target_sr),
        "-",
    ]
    out = _run_ffmpeg(cmd, stdin_bytes=data).stdout
    audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    return audio, target_sr


class AudioNormalizer:
    def __init__(self, target_dB_FS: float = -25.0, eps: float = 1e-6):
        self.target_dB_FS = target_dB_FS
        self.eps = eps

    def tailor_dB_FS(self, audio: np.ndarray) -> tuple[np.ndarray, float, float]:
        rms = np.sqrt(np.mean(audio**2))
        scalar = 10 ** (self.target_dB_FS / 20) / (rms + self.eps)
        return audio * scalar, rms, scalar

    def avoid_clipping(
        self, audio: np.ndarray, scalar: Optional[float] = None
    ) -> tuple[np.ndarray, float]:
        if scalar is None:
            max_val = np.max(np.abs(audio))
            scalar = max_val + self.eps if max_val > 1.0 else 1.0
        return audio / scalar, scalar

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        audio, _, _ = self.tailor_dB_FS(audio)
        audio, _ = self.avoid_clipping(audio)
        return audio


class VibeVoiceASRTextTokenizerFast(Qwen2TokenizerFast):
    model_input_names = ["input_ids", "attention_mask"]
    _CHAT_TEMPLATE = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n'}}"
        "{% if message['content'] is string %}"
        "{{ message['content'] }}"
        "{% else %}"
        "{% for content in message['content'] %}"
        "{% if content['type'] == 'text' %}{{ content['text'] }}"
        "{% elif content['type'] == 'input_audio' %}{{ '<|speech_start|><|speech_pad|><|speech_end|>' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "{{ '<|im_end|>\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        self._apply_vibevoice_extensions()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        base_tokenizer = Qwen2TokenizerFast.from_pretrained(
            pretrained_model_name_or_path,
            *inputs,
            **kwargs,
        )
        base_tokenizer.__class__ = cls
        base_tokenizer._apply_vibevoice_extensions()
        return base_tokenizer

    def _apply_vibevoice_extensions(self):
        self._add_vibevoice_special_tokens()
        self.chat_template = self._CHAT_TEMPLATE

    def _add_vibevoice_special_tokens(self):
        self.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<|object_ref_start|>",
                    "<|object_ref_end|>",
                    "<|box_start|>",
                ]
            }
        )
        self._speech_start_id = self.convert_tokens_to_ids("<|object_ref_start|>")
        self._speech_end_id = self.convert_tokens_to_ids("<|object_ref_end|>")
        self._speech_pad_id = self.convert_tokens_to_ids("<|box_start|>")
        self._eos_id = self.eos_token_id
        pad_id = self.convert_tokens_to_ids("<|image_pad|>")
        self._pad_id = self.pad_token_id if pad_id is None or pad_id < 0 else pad_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def speech_start_id(self) -> int:
        return self._speech_start_id

    @property
    def speech_end_id(self) -> int:
        return self._speech_end_id

    @property
    def speech_pad_id(self) -> int:
        return self._speech_pad_id

    @property
    def pad_id(self) -> int:
        return self._pad_id


class VibeVoiceTokenizerProcessor(FeatureExtractionMixin):
    model_input_names = ["audio"]

    def __init__(
        self,
        sampling_rate: int = AUDIO_SAMPLE_RATE,
        normalize_audio: bool = True,
        target_dB_FS: float = -25.0,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.normalize_audio = normalize_audio
        self.normalizer = (
            AudioNormalizer(target_dB_FS=target_dB_FS, eps=eps)
            if normalize_audio
            else None
        )
        self.feature_extractor_dict = {
            "sampling_rate": sampling_rate,
            "normalize_audio": normalize_audio,
            "target_dB_FS": target_dB_FS,
            "eps": eps,
        }

    def _ensure_mono(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio
        if audio.ndim == 2:
            if audio.shape[0] == 2:
                return np.mean(audio, axis=0)
            if audio.shape[1] == 2:
                return np.mean(audio, axis=1)
            if audio.shape[0] == 1:
                return audio.squeeze(0)
            if audio.shape[1] == 1:
                return audio.squeeze(1)
        raise ValueError(f"Unexpected audio shape: {audio.shape}")

    def _load_audio_from_path(self, audio_path: str) -> np.ndarray:
        file_ext = os.path.splitext(audio_path)[1].lower()
        if HAS_FFMPEG_UTILS and file_ext in COMMON_AUDIO_EXTS:
            audio, sr = load_audio_use_ffmpeg(
                audio_path, resample=True, target_sr=self.sampling_rate
            )
        else:
            import soundfile as sf
            from scipy.signal import resample

            audio, sr = sf.read(audio_path)
            if sr != self.sampling_rate:
                num_samples = int(len(audio) * float(self.sampling_rate) / sr)
                audio = resample(audio, num_samples)
        audio = self._ensure_mono(np.asarray(audio, dtype=np.float32))
        if self.normalizer is not None:
            audio = self.normalizer(audio)
        return audio

    def __call__(
        self,
        audio: Union[
            str,
            np.ndarray,
            list[float],
            list[np.ndarray],
            list[list[float]],
            list[str],
        ] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ):
        if audio is None:
            raise ValueError("Audio input is required")
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            warnings.warn(
                f"Input sampling rate {sampling_rate} differs from expected {self.sampling_rate}"
            )

        if isinstance(audio, str):
            processed = [self._load_audio_from_path(audio)]
        elif isinstance(audio, list) and audio and isinstance(audio[0], str):
            processed = [self._load_audio_from_path(path) for path in audio]
        elif isinstance(audio, list) and audio and isinstance(audio[0], (list, np.ndarray)):
            processed = [np.asarray(x, dtype=np.float32) for x in audio]
        else:
            processed = [np.asarray(audio, dtype=np.float32)]

        processed = [self._ensure_mono(x) for x in processed]
        if self.normalizer is not None:
            processed = [self.normalizer(x) for x in processed]

        if return_tensors == "pt":
            max_len = max(len(x) for x in processed)
            padded = np.zeros((len(processed), max_len), dtype=np.float32)
            lengths = []
            for i, audio_item in enumerate(processed):
                padded[i, : len(audio_item)] = audio_item
                lengths.append(len(audio_item))
            return {
                "audio": torch.from_numpy(padded),
                "audio_lengths": torch.tensor(lengths, dtype=torch.long),
            }

        return {"audio": processed}


class VibeVoiceASRProcessor:
    def __init__(
        self,
        tokenizer=None,
        audio_processor=None,
        speech_tok_compress_ratio: int = 3200,
        target_sample_rate: int = AUDIO_SAMPLE_RATE,
        normalize_audio: bool = True,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.speech_tok_compress_ratio = speech_tok_compress_ratio
        self.target_sample_rate = target_sample_rate
        self.normalize_audio = normalize_audio
        self.audio_normalizer = AudioNormalizer() if normalize_audio else None
        self._cache_special_tokens()

    def _cache_special_tokens(self):
        self.speech_start_id = getattr(
            self.tokenizer,
            "speech_start_id",
            self.tokenizer.convert_tokens_to_ids("<|object_ref_start|>"),
        )
        self.speech_end_id = getattr(
            self.tokenizer,
            "speech_end_id",
            self.tokenizer.convert_tokens_to_ids("<|object_ref_end|>"),
        )
        self.speech_pad_id = getattr(
            self.tokenizer,
            "speech_pad_id",
            self.tokenizer.convert_tokens_to_ids("<|box_start|>"),
        )
        self.pad_id = getattr(self.tokenizer, "pad_id", self.tokenizer.pad_token_id)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        processor_kwargs = dict(kwargs)
        processor_kwargs.pop("revision", None)
        processor_kwargs.pop("tokenizer_revision", None)
        local_path = pretrained_model_name_or_path
        if not os.path.isdir(local_path):
            local_path = snapshot_download(
                pretrained_model_name_or_path,
                allow_patterns=["preprocessor_config.json"],
            )
        config_path = os.path.join(local_path, "preprocessor_config.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

        language_model_name = config.get("language_model_pretrained_name") or processor_kwargs.pop(
            "language_model_pretrained_name",
            "Qwen/Qwen2.5-1.5B",
        )
        tokenizer = VibeVoiceASRTextTokenizerFast.from_pretrained(
            language_model_name,
            **processor_kwargs,
        )
        audio_processor = VibeVoiceTokenizerProcessor(
            sampling_rate=config.get("target_sample_rate", AUDIO_SAMPLE_RATE),
            normalize_audio=config.get("normalize_audio", True),
            target_dB_FS=config.get("target_dB_FS", -25.0),
            eps=config.get("eps", 1e-6),
        )
        return cls(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
            speech_tok_compress_ratio=config.get("speech_tok_compress_ratio", 3200),
            target_sample_rate=config.get("target_sample_rate", AUDIO_SAMPLE_RATE),
            normalize_audio=config.get("normalize_audio", True),
        )


class ConvLayerNorm(nn.LayerNorm):
    def forward(self, x):
        x = x.transpose(1, 2)
        x = nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float(),
            self.bias.float(),
            self.eps,
        ).type_as(x)
        return x.transpose(1, 2)


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        weight_shape=None,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            weight_shape = (dim,) if weight_shape is None else weight_shape
            self.weight = nn.Parameter(torch.ones(weight_shape))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


class ConvRMSNorm(RMSNorm):
    def forward(self, x):
        x = x.transpose(1, 2)
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output.transpose(1, 2)


CONV_NORMALIZATIONS = frozenset(
    [
        "none",
        "weight_norm",
        "spectral_norm",
        "time_layer_norm",
        "layer_norm",
        "time_group_norm",
    ]
)


def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    if norm == "weight_norm":
        return nn.utils.weight_norm(module)
    if norm == "spectral_norm":
        return nn.utils.spectral_norm(module)
    return module


def get_norm_module(
    module: nn.Module, causal: bool = False, norm: str = "none", **norm_kwargs
) -> nn.Module:
    if norm == "layer_norm":
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    if norm == "time_group_norm":
        if causal:
            raise ValueError("GroupNorm does not support causal evaluation.")
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    return nn.Identity()


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return int(ideal_length - length)


def pad1d(
    x: torch.Tensor, paddings: Tuple[int, int], mode: str = "constant", value: float = 0.0
):
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: Tuple[int, int]):
    padding_left, padding_right = paddings
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class NormConv1d(nn.Module):
    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)

    def forward(self, x):
        return self.norm(self.conv(x))


class NormConvTranspose1d(nn.Module):
    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)

    def forward(self, x):
        return self.norm(self.convtr(x))


class SConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        causal: bool = False,
        pad_mode: str = "constant",
        norm: str = "none",
        norm_kwargs: Dict[str, Any] | None = None,
        bias: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.pad_mode = pad_mode
        norm_kwargs = norm_kwargs or {}
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(
        self,
        x,
        cache=None,
        sample_indices=None,
        use_cache=False,
        debug=False,
        is_final_chunk=False,
    ):
        kernel_size = (self.kernel_size - 1) * self.dilation + 1
        padding_total = kernel_size - self.stride
        extra_padding = get_extra_padding_for_conv1d(
            x, kernel_size, self.stride, padding_total=padding_total
        )

        if self.causal:
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            right = padding_total // 2
            left = padding_total - right
            x = pad1d(x, (left, right + extra_padding), mode=self.pad_mode)
        return self.conv(x)


class SConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        causal: bool = False,
        pad_mode: str = "constant",
        norm: str = "none",
        norm_kwargs: Dict[str, Any] | None = None,
        trim_right_ratio: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        norm_kwargs = norm_kwargs or {}
        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.pad_mode = pad_mode
        self.trim_right_ratio = trim_right_ratio
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(
        self,
        x,
        cache=None,
        sample_indices=None,
        use_cache=False,
        debug=False,
        is_final_chunk=False,
    ):
        y = self.convtr(x)
        padding_total = self.kernel_size - self.stride
        if self.causal:
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
        return unpad1d(y, (padding_left, padding_right))


class FFN(nn.Module):
    def __init__(self, embed_dim, ffn_dim, bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(self.embed_dim, ffn_dim, bias=bias)
        self.gelu = ACT2FN["gelu"]
        self.linear2 = nn.Linear(ffn_dim, self.embed_dim, bias=bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class Convlayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        pad_mode="zeros",
        norm="weight_norm",
        causal=True,
    ):
        super().__init__()
        self.conv = SConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            pad_mode=pad_mode,
            norm=norm,
            causal=causal,
        )

    def forward(self, x):
        return self.conv(x)


class Block1D(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=7,
        drop_path=0.0,
        mixer_layer="conv",
        layer_scale_init_value=1e-6,
        **kwargs,
    ):
        super().__init__()

        if kwargs.get("layernorm", "LN") == "LN":
            self.norm = ConvLayerNorm(dim, eps=kwargs.get("eps", 1e-6))
            self.ffn_norm = ConvLayerNorm(dim, eps=kwargs.get("eps", 1e-6))
        elif kwargs.get("layernorm", "RMSNorm") == "RMSNorm":
            self.norm = ConvRMSNorm(
                dim,
                eps=kwargs.get("eps", 1e-6),
                elementwise_affine=kwargs.get("layernorm_elementwise_affine", True),
            )
            self.ffn_norm = ConvRMSNorm(
                dim,
                eps=kwargs.get("eps", 1e-6),
                elementwise_affine=kwargs.get("layernorm_elementwise_affine", True),
            )
        else:
            raise ValueError(f"Unsupported layernorm: {kwargs.get('layernorm')}")

        if mixer_layer == "conv":
            self.mixer = Convlayer(
                dim,
                dim,
                groups=kwargs.get("groups", 1),
                kernel_size=kernel_size,
                pad_mode=kwargs.get("pad_mode", "reflect"),
                norm=kwargs.get("norm", "none"),
                causal=kwargs.get("causal", True),
                bias=kwargs.get("bias", True),
            )
        elif mixer_layer == "depthwise_conv":
            self.mixer = Convlayer(
                dim,
                dim,
                groups=dim,
                kernel_size=kernel_size,
                pad_mode=kwargs.get("pad_mode", "reflect"),
                norm=kwargs.get("norm", "none"),
                causal=kwargs.get("causal", True),
                bias=kwargs.get("bias", True),
            )
        else:
            raise ValueError(f"Unsupported mixer layer: {mixer_layer}")

        self.ffn = FFN(
            dim,
            kwargs.get("ffn_expansion", 4) * dim,
            bias=kwargs.get("bias", False),
        )
        self.drop_path = nn.Identity()
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
            self.ffn_gamma = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma = None
            self.ffn_gamma = None

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        if self.gamma is not None:
            x = x * self.gamma.unsqueeze(-1)
        x = residual + self.drop_path(x)

        residual = x
        x = self.ffn_norm(x)
        x = x.permute(0, 2, 1)
        x = self.ffn(x)
        x = x.permute(0, 2, 1)
        if self.ffn_gamma is not None:
            x = x * self.ffn_gamma.unsqueeze(-1)
        x = residual + self.drop_path(x)
        return x


class TokenizerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.channels = config.channels
        self.dimension = config.dimension
        self.n_filters = config.n_filters
        self.ratios = list(reversed(config.ratios))
        self.depths = config.depths
        self.n_residual_layers = getattr(config, "n_residual_layers", 1)
        self.hop_length = np.prod(self.ratios)
        self.causal = config.causal

        kernel_size = getattr(config, "kernel_size", 7)
        last_kernel_size = getattr(config, "last_kernel_size", 7)
        norm = getattr(config, "norm", "none")
        norm_params = getattr(config, "norm_params", {})
        pad_mode = getattr(config, "pad_mode", "reflect")
        bias = getattr(config, "bias", True)
        layernorm = getattr(config, "layernorm", "LN")
        layernorm_eps = getattr(config, "layernorm_eps", 1e-6)
        layernorm_elementwise_affine = getattr(
            config, "layernorm_elementwise_affine", True
        )
        drop_path_rate = getattr(config, "drop_path_rate", 0.0)
        mixer_layer = getattr(config, "mixer_layer", "conv")
        layer_scale_init_value = getattr(config, "layer_scale_init_value", 0)
        disable_last_norm = getattr(config, "disable_last_norm", False)

        if layernorm == "LN":
            norm_type = ConvLayerNorm
        elif layernorm == "RMSNorm":
            norm_type = partial(
                ConvRMSNorm,
                elementwise_affine=layernorm_elementwise_affine,
            )
        else:
            raise ValueError(f"Unsupported norm type: {layernorm}")

        stem = nn.Sequential(
            SConv1d(
                self.channels,
                self.n_filters,
                kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=self.causal,
                pad_mode=pad_mode,
                bias=bias,
            ),
        )

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(stem)
        for i in range(len(self.ratios)):
            in_ch = self.n_filters * (2**i)
            out_ch = self.n_filters * (2 ** (i + 1))
            downsample_layer = nn.Sequential(
                SConv1d(
                    in_ch,
                    out_ch,
                    kernel_size=self.ratios[i] * 2,
                    stride=self.ratios[i],
                    causal=self.causal,
                    pad_mode=pad_mode,
                    norm=norm,
                    bias=bias,
                )
            )
            self.downsample_layers.append(downsample_layer)

        layer_type = partial(
            Block1D,
            mixer_layer=mixer_layer,
            layernorm=layernorm,
            layernorm_elementwise_affine=layernorm_elementwise_affine,
            eps=layernorm_eps,
            causal=self.causal,
            pad_mode=pad_mode,
            norm=norm,
            bias=bias,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(len(self.depths)):
            in_ch = self.n_filters * (2**i)
            stage = nn.Sequential(
                *[layer_type(dim=in_ch, drop_path=dp_rates[cur + j]) for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]

        if not disable_last_norm:
            self.norm = norm_type(in_ch, eps=layernorm_eps)
        else:
            self.norm = nn.Identity()
        self.head = SConv1d(
            in_ch,
            self.dimension,
            kernel_size=last_kernel_size,
            causal=self.causal,
            pad_mode=pad_mode,
            norm=norm,
            bias=bias,
        )

    def forward_features(
        self, x, cache=None, sample_indices=None, use_cache=False, debug=False, is_final_chunk=False
    ):
        for i in range(len(self.depths)):
            for layer in self.downsample_layers[i]:
                if isinstance(layer, SConv1d):
                    x = layer(
                        x,
                        cache=cache,
                        sample_indices=sample_indices,
                        use_cache=use_cache,
                        debug=debug,
                    )
                else:
                    x = layer(x)

            for block in self.stages[i]:
                if (
                    hasattr(block, "mixer")
                    and hasattr(block.mixer, "conv")
                    and isinstance(block.mixer.conv, SConv1d)
                ):
                    residual = x
                    x = block.norm(x)
                    x = block.mixer.conv(
                        x,
                        cache=cache,
                        sample_indices=sample_indices,
                        use_cache=use_cache,
                        debug=debug,
                    )
                    if block.gamma is not None:
                        x = x * block.gamma.unsqueeze(-1)
                    x = residual + x

                    residual = x
                    x = block.ffn_norm(x)
                    x = x.permute(0, 2, 1)
                    x = block.ffn(x)
                    x = x.permute(0, 2, 1)
                    if block.ffn_gamma is not None:
                        x = x * block.ffn_gamma.unsqueeze(-1)
                    x = residual + x
                else:
                    x = block(x)

        return self.norm(x)

    def forward(
        self, x, cache=None, sample_indices=None, use_cache=False, debug=False, is_final_chunk=False
    ):
        x = self.forward_features(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
            is_final_chunk=is_final_chunk,
        )
        x = self.head(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )
        return x


class TokenizerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dimension = config.dimension
        self.channels = config.channels
        self.n_filters = config.n_filters
        self.ratios = config.ratios
        self.depths = config.depths
        self.n_residual_layers = getattr(config, "n_residual_layers", 1)
        self.hop_length = np.prod(self.ratios)
        self.causal = config.causal

        kernel_size = getattr(config, "kernel_size", 7)
        last_kernel_size = getattr(config, "last_kernel_size", 7)
        norm = getattr(config, "norm", "none")
        norm_params = getattr(config, "norm_params", {})
        pad_mode = getattr(config, "pad_mode", "reflect")
        bias = getattr(config, "bias", True)
        layernorm = getattr(config, "layernorm", "LN")
        layernorm_eps = getattr(config, "layernorm_eps", 1e-6)
        trim_right_ratio = getattr(config, "trim_right_ratio", 1.0)
        layernorm_elementwise_affine = getattr(
            config, "layernorm_elementwise_affine", True
        )
        drop_path_rate = getattr(config, "drop_path_rate", 0.0)
        mixer_layer = getattr(config, "mixer_layer", "conv")
        layer_scale_init_value = getattr(config, "layer_scale_init_value", 0)
        disable_last_norm = getattr(config, "disable_last_norm", False)

        if layernorm == "LN":
            norm_type = ConvLayerNorm
        elif layernorm == "RMSNorm":
            norm_type = partial(
                ConvRMSNorm,
                elementwise_affine=layernorm_elementwise_affine,
            )
        else:
            raise ValueError(f"Unsupported norm type: {layernorm}")

        stem = nn.Sequential(
            SConv1d(
                self.dimension,
                self.n_filters * 2 ** (len(self.depths) - 1),
                kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=self.causal,
                pad_mode=pad_mode,
                bias=bias,
            ),
        )

        self.upsample_layers = nn.ModuleList()
        self.upsample_layers.append(stem)
        for i in range(len(self.ratios)):
            in_ch = self.n_filters * (2 ** (len(self.depths) - 1 - i))
            out_ch = self.n_filters * (2 ** (len(self.depths) - 1 - i - 1))
            upsample_layer = nn.Sequential(
                SConvTranspose1d(
                    in_ch,
                    out_ch,
                    kernel_size=self.ratios[i] * 2,
                    stride=self.ratios[i],
                    norm=norm,
                    norm_kwargs=norm_params,
                    bias=bias,
                    causal=self.causal,
                    trim_right_ratio=trim_right_ratio,
                ),
            )
            self.upsample_layers.append(upsample_layer)

        layer_type = partial(
            Block1D,
            mixer_layer=mixer_layer,
            layernorm=layernorm,
            layernorm_elementwise_affine=layernorm_elementwise_affine,
            eps=layernorm_eps,
            causal=self.causal,
            pad_mode=pad_mode,
            norm=norm,
            bias=bias,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(len(self.depths)):
            in_ch = self.n_filters * (2 ** (len(self.depths) - 1 - i))
            stage = nn.Sequential(
                *[layer_type(dim=in_ch, drop_path=dp_rates[cur + j]) for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]

        if not disable_last_norm:
            self.norm = norm_type(in_ch, eps=layernorm_eps)
        else:
            self.norm = nn.Identity()
        self.head = SConv1d(
            in_ch,
            self.channels,
            kernel_size=last_kernel_size,
            causal=self.causal,
            pad_mode=pad_mode,
            norm=norm,
            bias=bias,
        )

    def forward_features(self, x, cache=None, sample_indices=None, use_cache=False, debug=False):
        for i in range(len(self.depths)):
            for layer in self.upsample_layers[i]:
                if isinstance(layer, (SConv1d, SConvTranspose1d)):
                    x = layer(
                        x,
                        cache=cache,
                        sample_indices=sample_indices,
                        use_cache=use_cache,
                        debug=debug,
                    )
                else:
                    x = layer(x)

            for block in self.stages[i]:
                if (
                    hasattr(block, "mixer")
                    and hasattr(block.mixer, "conv")
                    and isinstance(block.mixer.conv, SConv1d)
                ):
                    residual = x
                    x = block.norm(x)
                    x = block.mixer.conv(
                        x,
                        cache=cache,
                        sample_indices=sample_indices,
                        use_cache=use_cache,
                        debug=debug,
                    )
                    if block.gamma is not None:
                        x = x * block.gamma.unsqueeze(-1)
                    x = residual + x

                    residual = x
                    x = block.ffn_norm(x)
                    x = x.permute(0, 2, 1)
                    x = block.ffn(x)
                    x = x.permute(0, 2, 1)
                    if block.ffn_gamma is not None:
                        x = x * block.ffn_gamma.unsqueeze(-1)
                    x = residual + x
                else:
                    x = block(x)

        return self.norm(x)

    def forward(self, x, cache=None, sample_indices=None, use_cache=False, debug=False):
        x = self.forward_features(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )
        x = self.head(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )
        return x


class VibeVoiceTokenizerStreamingCache:
    def __init__(self):
        self.cache = {}

    def get(self, layer_id: str, sample_indices: torch.Tensor) -> Optional[torch.Tensor]:
        states = []
        max_length = 0
        for idx in sample_indices.tolist():
            key = (layer_id, idx)
            if key not in self.cache:
                return None
            state = self.cache[key]
            states.append(state)
            max_length = max(max_length, state.shape[-1])
        padded_states = []
        for state in states:
            if state.shape[-1] < max_length:
                pad_size = max_length - state.shape[-1]
                state = F.pad(state, (pad_size, 0), mode="constant", value=0)
            padded_states.append(state)
        return torch.stack(padded_states, dim=0)

    def set(self, layer_id: str, sample_indices: torch.Tensor, states: torch.Tensor):
        for i, idx in enumerate(sample_indices.tolist()):
            self.cache[(layer_id, idx)] = states[i].detach()

    def clear(self, layer_id: Optional[str] = None, sample_indices: Optional[torch.Tensor] = None):
        if layer_id is None and sample_indices is None:
            self.cache.clear()
            return
        keys_to_remove = []
        for layer_name, sample_idx in self.cache.keys():
            if layer_id is not None and layer_name != layer_id:
                continue
            if sample_indices is not None and sample_idx not in sample_indices.tolist():
                continue
            keys_to_remove.append((layer_name, sample_idx))
        for key in keys_to_remove:
            del self.cache[key]


@dataclass
class VibeVoiceTokenizerEncoderOutput:
    mean: torch.Tensor
    std: Optional[Union[float, torch.Tensor]] = None

    def sample(self, dist_type: str = "fix"):
        if dist_type == "fix":
            return self.mean + self.std * torch.randn_like(self.mean), self.std
        if dist_type == "gaussian":
            batch_size = self.mean.size(0)
            value = self.std / 0.8
            std = torch.randn(batch_size, device=self.mean.device, dtype=self.mean.dtype) * value
            while std.dim() < self.mean.dim():
                std = std.unsqueeze(-1)
            return self.mean + std * torch.randn_like(self.mean), std
        return self.mean, self.std

    def mode(self):
        return self.mean


class VibeVoiceAcousticTokenizerModel(PreTrainedModel):
    config_class = None
    base_model_prefix = "vibevoice_acoustic_tokenizer"

    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("fix_std", torch.tensor(config.fix_std), persistent=False)
        self.std_dist_type = getattr(config, "std_dist_type", "fix")
        encoder_depths = (
            [int(d) for d in config.encoder_depths.split("-")]
            if isinstance(config.encoder_depths, str)
            else config.encoder_depths
        )
        if config.decoder_depths is not None and isinstance(config.decoder_depths, str):
            decoder_depths = [int(d) for d in config.decoder_depths.split("-")]
        else:
            decoder_depths = list(reversed(encoder_depths))

        encoder_config = copy.deepcopy(config)
        encoder_config.dimension = config.vae_dim
        encoder_config.n_filters = config.encoder_n_filters
        encoder_config.ratios = config.encoder_ratios
        encoder_config.depths = encoder_depths
        encoder_config.norm = config.conv_norm
        encoder_config.pad_mode = config.pad_mode
        encoder_config.bias = config.conv_bias
        encoder_config.layernorm_eps = config.layernorm_eps
        encoder_config.layernorm_elementwise_affine = config.layernorm_elementwise_affine
        encoder_config.layer_scale_init_value = config.layer_scale_init_value
        encoder_config.disable_last_norm = config.disable_last_norm

        decoder_config = copy.deepcopy(config)
        decoder_config.dimension = config.vae_dim
        decoder_config.n_filters = config.decoder_n_filters
        decoder_config.ratios = config.decoder_ratios
        decoder_config.depths = decoder_depths
        decoder_config.norm = config.conv_norm
        decoder_config.pad_mode = config.pad_mode
        decoder_config.bias = config.conv_bias
        decoder_config.layernorm_eps = config.layernorm_eps
        decoder_config.layernorm_elementwise_affine = config.layernorm_elementwise_affine
        decoder_config.layer_scale_init_value = config.layer_scale_init_value
        decoder_config.disable_last_norm = config.disable_last_norm

        self.encoder = TokenizerEncoder(encoder_config)
        self.decoder = TokenizerDecoder(decoder_config)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @torch.no_grad()
    def encode(self, audio, cache=None, sample_indices=None, use_cache=False, debug=False, is_final_chunk=False):
        latents = self.encoder(
            audio,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
            is_final_chunk=is_final_chunk,
        )
        return VibeVoiceTokenizerEncoderOutput(mean=latents.permute(0, 2, 1), std=self.fix_std)

    @torch.no_grad()
    def sampling(self, encoder_output, dist_type=None):
        dist_type = dist_type or self.std_dist_type
        if dist_type in {"fix", "gaussian"}:
            return encoder_output.sample(dist_type=dist_type)
        raise ValueError(f"Unsupported dist_type: {dist_type}")

    @torch.no_grad()
    def decode(self, latents, cache=None, sample_indices=None, use_cache=False, debug=False):
        if latents.shape[1] != self.config.vae_dim:
            latents = latents.permute(0, 2, 1)
        return self.decoder(
            latents,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )


class VibeVoiceSemanticTokenizerModel(PreTrainedModel):
    config_class = None
    base_model_prefix = "vibevoice_semantic_tokenizer"

    def __init__(self, config):
        super().__init__(config)
        encoder_depths = (
            [int(d) for d in config.encoder_depths.split("-")]
            if isinstance(config.encoder_depths, str)
            else config.encoder_depths
        )
        encoder_config = copy.deepcopy(config)
        encoder_config.dimension = config.vae_dim
        encoder_config.n_filters = config.encoder_n_filters
        encoder_config.ratios = config.encoder_ratios
        encoder_config.depths = encoder_depths
        encoder_config.norm = config.conv_norm
        encoder_config.pad_mode = config.pad_mode
        encoder_config.bias = config.conv_bias
        encoder_config.layernorm_eps = config.layernorm_eps
        encoder_config.layernorm_elementwise_affine = config.layernorm_elementwise_affine
        encoder_config.layer_scale_init_value = config.layer_scale_init_value
        encoder_config.disable_last_norm = config.disable_last_norm
        self.encoder = TokenizerEncoder(encoder_config)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @torch.no_grad()
    def encode(self, audio, cache=None, sample_indices=None, use_cache=False, debug=False, is_final_chunk=False):
        latents = self.encoder(
            audio,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
            is_final_chunk=is_final_chunk,
        )
        return VibeVoiceTokenizerEncoderOutput(mean=latents.permute(0, 2, 1))
