import logging
import math
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.vibevoice_asr import (
    VibeVoiceASRForConditionalGeneration,
    VibeVoiceForASRTraining,
)
from sglang.srt.models.vibevoice_asr_utils import (
    AUDIO_SAMPLE_RATE,
    AudioNormalizer,
    load_audio_data_url_fast,
    load_audio_bytes_use_fast_wav,
    load_audio_bytes_use_ffmpeg,
    load_audio_use_ffmpeg,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import load_audio

logger = logging.getLogger(__name__)


class VibeVoiceASRMultimodalProcessor(BaseMultimodalProcessor):
    models = [VibeVoiceASRForConditionalGeneration, VibeVoiceForASRTraining]

    def __init__(
        self, hf_config, server_args, _processor, transport_mode, *args, **kwargs
    ):
        super().__init__(hf_config, server_args, _processor, transport_mode, *args, **kwargs)
        self.tokenizer = self._processor.tokenizer
        self.compress_ratio = int(getattr(hf_config, "speech_tok_compress_ratio", 3200))
        self.target_sample_rate = int(
            getattr(hf_config, "target_sample_rate", AUDIO_SAMPLE_RATE)
        )
        self.normalizer = AudioNormalizer()

        self.speech_start_id = getattr(self.tokenizer, "speech_start_id")
        self.speech_pad_id = getattr(self.tokenizer, "speech_pad_id")
        self.speech_end_id = getattr(self.tokenizer, "speech_end_id")

        self.speech_start_token = self.tokenizer.convert_ids_to_tokens(
            self.speech_start_id
        )
        self.speech_pad_token = self.tokenizer.convert_ids_to_tokens(self.speech_pad_id)
        self.speech_end_token = self.tokenizer.convert_ids_to_tokens(self.speech_end_id)

        # Define the audio placeholder token for SGLang's multimodal pipeline.
        # This tells SGLang what to insert in the prompt text when it sees an audio_url.
        audio_token = (
            self.speech_start_token
            + self.speech_pad_token
            + self.speech_end_token
        )
        audio_token_regex = re.compile(
            re.escape(self.speech_start_token)
            + "(?:" + re.escape(self.speech_pad_token) + ")+"
            + re.escape(self.speech_end_token)
        )
        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=audio_token,
            audio_token_regex=audio_token_regex,
            audio_token_id=self.speech_pad_id,
        ).build(_processor)

    def _load_audio(self, audio_item) -> np.ndarray:
        if isinstance(audio_item, bytes):
            try:
                audio, _ = load_audio_bytes_use_fast_wav(
                    audio_item,
                    target_sr=self.target_sample_rate,
                )
            except Exception:
                audio, _ = load_audio_bytes_use_ffmpeg(
                    audio_item, resample=True, target_sr=self.target_sample_rate
                )
        elif isinstance(audio_item, str):
            if audio_item.startswith("data:"):
                try:
                    audio, _ = load_audio_data_url_fast(
                        audio_item, target_sr=self.target_sample_rate
                    )
                except Exception as exc:
                    logger.warning(
                        "VibeVoice fast data-url audio decode fell back to generic loader: %s",
                        exc,
                    )
                    audio = load_audio(audio_item, sr=self.target_sample_rate)
            else:
                try:
                    audio, _ = load_audio_use_ffmpeg(
                        audio_item, resample=True, target_sr=self.target_sample_rate
                    )
                except Exception:
                    audio = load_audio(audio_item, sr=self.target_sample_rate)
        else:
            audio = np.asarray(audio_item, dtype=np.float32)
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        audio = self.normalizer(audio)
        return audio

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj=None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        if not audio_data:
            return None
        if len(audio_data) != 1:
            raise ValueError(
                f"VibeVoice expects exactly 1 audio input, got {len(audio_data)}"
            )

        audio = self._load_audio(audio_data[0])
        num_features = max(1, int(math.ceil(len(audio) / self.compress_ratio)))

        # Build the speech token sequence for this audio's length
        speech_tokens = (
            self.speech_start_token
            + self.speech_pad_token * num_features
            + self.speech_end_token
        )

        # The server inserts a single audio_token placeholder in input_text.
        # Replace it with the correct number of speech pad tokens for this audio.
        placeholder_found = False
        if self.mm_tokens.audio_token_regex is not None:
            modified_text = self.mm_tokens.audio_token_regex.sub(
                speech_tokens, input_text, count=1,
            )
            placeholder_found = (modified_text != input_text)
        else:
            modified_text = input_text

        if not placeholder_found:
            # Fallback: if no placeholder found, inject after user turn marker
            user_marker = "<|im_start|>user\n"
            pos = input_text.find(user_marker)
            if pos >= 0:
                insert_pos = pos + len(user_marker)
                modified_text = (
                    input_text[:insert_pos]
                    + speech_tokens + "\n"
                    + input_text[insert_pos:]
                )
            else:
                modified_text = speech_tokens + "\n" + input_text

        input_ids = self.tokenizer.encode(modified_text, add_special_tokens=False)
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        audio_offsets = self.get_mm_items_offset(
            input_ids=input_ids_tensor,
            mm_token_id=self.speech_pad_id,
        )

        return {
            "input_ids": input_ids,
            "mm_items": [
                MultimodalDataItem(
                    modality=Modality.AUDIO,
                    feature=torch.from_numpy(audio),
                    offsets=audio_offsets,
                    model_specific_data={"audio_length": len(audio)},
                )
            ],
            "audio_start_id": self.speech_start_id,
            "audio_token_id": self.speech_pad_id,
            "audio_end_id": self.speech_end_id,
        }
