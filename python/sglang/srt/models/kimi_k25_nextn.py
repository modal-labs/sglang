"""Inference-only Kimi-K2.5 NextN speculative decoding model."""

from typing import Iterable, Optional, Tuple

import torch
from transformers import PretrainedConfig

from sglang.srt.configs.kimi_k25 import KimiK25Config
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from sglang.srt.models.utils import WeightsMapper


class KimiK25ForConditionalGenerationNextN(DeepseekV3ForCausalLMNextN):
    # Support nvidia/Kimi-K2.5-NVFP4 naming: language_model.layers.*.
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.layers.": "language_model.model.layers.",
        }
    )

    def __init__(
        self,
        config: KimiK25Config | PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Draft uses only the text/NextN part of Kimi config.
        text_config = config.text_config if hasattr(config, "text_config") else config
        super().__init__(text_config, quant_config=quant_config, prefix=prefix)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        mapper = getattr(self, "hf_to_sglang_mapper", None)
        if mapper is not None:
            weights = mapper.apply(weights)

        language_weights = []
        for name, loaded_weight in weights:
            if "vision_tower" in name or "mm_projector" in name:
                continue
            if name.startswith("language_model."):
                name = name.replace("language_model.", "", 1)
            language_weights.append((name, loaded_weight))

        super().load_weights(language_weights)


EntryClass = [KimiK25ForConditionalGenerationNextN]
