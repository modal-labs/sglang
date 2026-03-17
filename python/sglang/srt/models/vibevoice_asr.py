import logging
import os
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from sglang.srt.configs.vibevoice_asr import (
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceConfig,
    VibeVoiceSemanticTokenizerConfig,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.models.vibevoice_asr_utils import (
    VibeVoiceAcousticTokenizerModel,
    VibeVoiceSemanticTokenizerModel,
    VibeVoiceTokenizerEncoderOutput,
    VibeVoiceTokenizerStreamingCache,
)
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class VibeVoiceASRRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class VibeVoiceASRSpeechConnector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = VibeVoiceASRRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.norm(self.fc1(x)))


class VibeVoiceAudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        def get_cfg(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        self.acoustic_vae_dim = get_cfg(config, "acoustic_vae_dim", 64)
        self.semantic_vae_dim = get_cfg(config, "semantic_vae_dim", 128)

        decoder_config = get_cfg(config, "decoder_config")
        target_hidden_size = get_cfg(decoder_config, "hidden_size")
        if target_hidden_size is None:
            target_hidden_size = get_cfg(config, "hidden_size", 3584)
        self.hidden_size = target_hidden_size

        ac_cfg = get_cfg(config, "acoustic_tokenizer_config")
        sc_cfg = get_cfg(config, "semantic_tokenizer_config")
        if isinstance(ac_cfg, dict):
            acoustic_config = VibeVoiceAcousticTokenizerConfig(**ac_cfg)
        else:
            acoustic_config = ac_cfg
        if isinstance(sc_cfg, dict):
            semantic_config = VibeVoiceSemanticTokenizerConfig(**sc_cfg)
        else:
            semantic_config = sc_cfg

        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(acoustic_config)
        self.semantic_tokenizer = VibeVoiceSemanticTokenizerModel(semantic_config)

        root_torch_dtype = get_cfg(config, "torch_dtype", None)
        if isinstance(root_torch_dtype, str):
            self._audio_encoder_dtype = getattr(torch, root_torch_dtype)
        else:
            self._audio_encoder_dtype = root_torch_dtype or torch.float32

        self.acoustic_connector = VibeVoiceASRSpeechConnector(
            self.acoustic_vae_dim, self.hidden_size
        )
        self.semantic_connector = VibeVoiceASRSpeechConnector(
            self.semantic_vae_dim, self.hidden_size
        )
        self.compress_ratio = get_cfg(config, "speech_tok_compress_ratio", 3200)
        self.sample_rate = get_cfg(config, "target_sample_rate", 24000)
        self.enable_streaming = get_cfg(config, "enable_streaming", True)
        self.streaming_segment_duration = get_cfg(
            config, "streaming_segment_duration", 60.0
        )
        use_mean_env = os.getenv("VIBEVOICE_USE_MEAN", "").strip().lower()
        self.use_sample = use_mean_env not in ("1", "true", "yes")
        self._lm_dtype: torch.dtype = torch.bfloat16

    def _ensure_audio_encoder_dtype(self):
        target_dtype = self._audio_encoder_dtype
        for module_name in (
            "acoustic_tokenizer",
            "semantic_tokenizer",
            "acoustic_connector",
            "semantic_connector",
        ):
            module = getattr(self, module_name)
            try:
                current_dtype = next(module.parameters()).dtype
            except StopIteration:
                continue
            if current_dtype != target_dtype:
                setattr(self, module_name, module.to(dtype=target_dtype))

    def forward(
        self,
        audio: torch.Tensor,
        *,
        use_streaming: bool = True,
        segment_duration_s: float | None = None,
        use_sample: bool | None = None,
    ) -> torch.Tensor:
        self._ensure_audio_encoder_dtype()
        audio = audio.to(dtype=self._audio_encoder_dtype)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        segment_duration = segment_duration_s or self.streaming_segment_duration
        segment_samples = int(segment_duration * self.sample_rate)
        use_streaming = (
            use_streaming
            and self.enable_streaming
            and audio.shape[-1] > segment_samples
        )
        if use_sample is None:
            use_sample = self.use_sample

        with torch.no_grad():
            if not use_streaming:
                acoustic_input = audio.unsqueeze(1)
                acoustic_out = self.acoustic_tokenizer.encode(acoustic_input)
                if use_sample:
                    acoustic_tokens = acoustic_out.sample(
                        dist_type=self.acoustic_tokenizer.std_dist_type
                    )[0]
                else:
                    acoustic_tokens = acoustic_out.mean
                acoustic_embeds = self.acoustic_connector(acoustic_tokens)

                semantic_out = self.semantic_tokenizer.encode(acoustic_input)
                semantic_embeds = self.semantic_connector(semantic_out.mean)
            else:
                acoustic_cache = VibeVoiceTokenizerStreamingCache()
                semantic_cache = VibeVoiceTokenizerStreamingCache()
                acoustic_segments = []
                semantic_segments = []
                batch_size = audio.shape[0]
                sample_indices = torch.arange(batch_size, device=audio.device)
                total_samples = audio.shape[-1]
                segments = [
                    (start, min(start + segment_samples, total_samples))
                    for start in range(0, total_samples, segment_samples)
                ]
                for seg_idx, (start, end) in enumerate(segments):
                    chunk = audio[:, start:end].contiguous()
                    if chunk.numel() == 0:
                        continue
                    is_final = seg_idx == len(segments) - 1
                    acoustic_enc = self.acoustic_tokenizer.encode(
                        chunk.unsqueeze(1),
                        cache=acoustic_cache,
                        sample_indices=sample_indices,
                        use_cache=True,
                        is_final_chunk=is_final,
                    )
                    semantic_enc = self.semantic_tokenizer.encode(
                        chunk.unsqueeze(1),
                        cache=semantic_cache,
                        sample_indices=sample_indices,
                        use_cache=True,
                        is_final_chunk=is_final,
                    )
                    acoustic_segments.append(acoustic_enc.mean)
                    semantic_segments.append(semantic_enc.mean)

                acoustic_mean = (
                    torch.cat(acoustic_segments, dim=1).contiguous()
                    if acoustic_segments
                    else torch.zeros(
                        (batch_size, 0, self.acoustic_vae_dim),
                        device=audio.device,
                        dtype=self._audio_encoder_dtype,
                    )
                )
                acoustic_full = VibeVoiceTokenizerEncoderOutput(
                    mean=acoustic_mean,
                    std=self.acoustic_tokenizer.fix_std,
                )
                if use_sample:
                    acoustic_tokens = acoustic_full.sample(
                        dist_type=self.acoustic_tokenizer.std_dist_type
                    )[0]
                else:
                    acoustic_tokens = acoustic_full.mean
                acoustic_embeds = self.acoustic_connector(acoustic_tokens)

                semantic_tokens = (
                    torch.cat(semantic_segments, dim=1).contiguous()
                    if semantic_segments
                    else torch.zeros(
                        (batch_size, 0, self.semantic_vae_dim),
                        device=audio.device,
                        dtype=self._audio_encoder_dtype,
                    )
                )
                semantic_embeds = self.semantic_connector(semantic_tokens)

        return (acoustic_embeds + semantic_embeds).to(dtype=self._lm_dtype)


class VibeVoiceASRForConditionalGeneration(nn.Module):
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: VibeVoiceConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.audio_encoder = VibeVoiceAudioEncoder(config)
        decoder_config = getattr(config, "decoder_config", config)
        self.language_model = Qwen2ForCausalLM(
            decoder_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        lm_dtype = getattr(decoder_config, "torch_dtype", None)
        if isinstance(lm_dtype, str):
            self.audio_encoder._lm_dtype = getattr(torch, lm_dtype)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        embeddings = []
        device = next(self.audio_encoder.parameters()).device
        for item in items:
            audio = item.feature
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)
            if not isinstance(audio, torch.Tensor):
                audio = torch.tensor(audio)
            audio = audio.to(device=device, dtype=torch.float32)
            if audio.ndim > 1:
                audio = audio.squeeze()
            actual_len = getattr(item, "audio_length", None)
            if actual_len is None:
                model_specific_data = getattr(item, "model_specific_data", None) or {}
                actual_len = model_specific_data.get("audio_length")
            if isinstance(actual_len, torch.Tensor):
                actual_len = int(actual_len.item())
            if actual_len:
                audio = audio[:actual_len]
            if audio.numel() < 160:
                continue
            embeddings.append(self.audio_encoder(audio).squeeze(0))
        if not embeddings:
            return torch.zeros(
                (0, self.config.decoder_config.hidden_size),
                device=device,
                dtype=self.audio_encoder._lm_dtype,
            )
        return torch.cat(embeddings, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={Modality.AUDIO: self.get_audio_feature},
            positions=positions,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        prefix_mapping = {
            "model.acoustic_tokenizer.": "audio_encoder.acoustic_tokenizer.",
            "model.semantic_tokenizer.": "audio_encoder.semantic_tokenizer.",
            "model.acoustic_connector.": "audio_encoder.acoustic_connector.",
            "model.semantic_connector.": "audio_encoder.semantic_connector.",
            "model.language_model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.language_model.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            for old_prefix, new_prefix in prefix_mapping.items():
                if name.startswith(old_prefix):
                    name = new_prefix + name[len(old_prefix) :]
                    break

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or "audio_encoder" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning("Parameter %s not found in params_dict", name)
                    continue
                param = params_dict[name]
                if param.shape != loaded_weight.shape and not hasattr(param, "weight_loader"):
                    logger.error(
                        "Shape mismatch for %s: checkpoint=%s model=%s",
                        name, list(loaded_weight.shape), list(param.shape),
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


    def set_eagle3_layers_to_capture(self, layer_ids=None):
        self.language_model.set_eagle3_layers_to_capture(layer_ids)

    def get_embed_and_head(self):
        return self.language_model.get_embed_and_head()

    def set_embed_and_head(self, embed, head):
        self.language_model.set_embed_and_head(embed, head)


class VibeVoiceForASRTraining(VibeVoiceASRForConditionalGeneration):
    pass


EntryClass = [VibeVoiceASRForConditionalGeneration, VibeVoiceForASRTraining]
