"""K3 image-prompt id expansion and the renderer-id passthrough gate (CPU)."""

import asyncio
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.multimodal.processors import kimi_k3
from sglang.srt.multimodal.processors.kimi_k3 import (
    KimiK3GPUProcessorWrapper,
    KimiK3ImageProcessor,
    _expand_k3_image_prompt_token_ids,
    _expand_k3_image_prompt_token_ids_vectorized,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

IMAGE_TOKEN_ID = 900
MEDIA_BEGIN = 901
MEDIA_CONTENT = 902
MEDIA_END = 903
SEP_CONTROL_ID = 904


class _StubTokenizer:
    """Char-per-token tokenizer whose special strings collapse to control ids
    only under ``allowed_special="all"`` -- the tiktoken contract K3 relies on."""

    SPECIALS = {
        "<|media_begin|>": MEDIA_BEGIN,
        "<|media_content|>": MEDIA_CONTENT,
        "<|media_end|>": MEDIA_END,
        "<|media_pad|>": IMAGE_TOKEN_ID,
        "<|sep|>": SEP_CONTROL_ID,
    }

    def encode(self, text, allowed_special=None):
        out = []
        i = 0
        while i < len(text):
            matched = None
            if allowed_special == "all" and text[i] == "<":
                for special, token_id in self.SPECIALS.items():
                    if text.startswith(special, i):
                        matched = (special, token_id)
                        break
            if matched is not None:
                out.append(matched[1])
                i += len(matched[0])
            else:
                out.append(ord(text[i]))
                i += 1
        return out

    def decode(self, ids):
        reverse = {v: k for k, v in self.SPECIALS.items()}
        return "".join(reverse.get(i, chr(i)) for i in ids)


def _expand_reference(input_ids, image_token_id, counts, sizes, tokenizer):
    """Per-token Python loop, kept as the identity oracle for the split path."""
    output = []
    image_index = 0
    for token_id in input_ids:
        if token_id != image_token_id:
            output.append(int(token_id))
            continue
        width, height = sizes[image_index]
        output.extend(
            tokenizer.encode(
                f"<|media_begin|>image {width}x{height}<|media_content|>",
                allowed_special="all",
            )
        )
        output.extend([image_token_id] * counts[image_index])
        output.extend(tokenizer.encode("<|media_end|>", allowed_special="all"))
        image_index += 1
    return output


class _Stop(Exception):
    pass


class _FastLoadCapture(KimiK3ImageProcessor):
    def __init__(self):
        self.mm_tokens = types.SimpleNamespace(image_token_id=IMAGE_TOKEN_ID)
        self.captured_input_ids = "unset"

    def validate_tokenized_image_placeholders(self, *_args):
        return True

    async def fast_load_mm_data(self, **kwargs):
        self.captured_input_ids = kwargs["input_ids"]
        raise _Stop()


PROMPTS = [
    [1, 2, IMAGE_TOKEN_ID, 3],
    [IMAGE_TOKEN_ID],
    [IMAGE_TOKEN_ID, IMAGE_TOKEN_ID],
    [IMAGE_TOKEN_ID, 4, 5],
    [5, IMAGE_TOKEN_ID, 6, 7, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 8],
    [1, 2, 3, IMAGE_TOKEN_ID],
]


def _case(prompt):
    n = torch.as_tensor(prompt).tolist().count(IMAGE_TOKEN_ID)
    counts = [3 + k for k in range(n)]
    sizes = [(64 + k, 32) for k in range(n)]
    return counts, sizes


def _make_wrapper(tok):
    wrapper = KimiK3GPUProcessorWrapper.__new__(KimiK3GPUProcessorWrapper)
    wrapper._hf_processor = types.SimpleNamespace(tokenizer=tok)
    wrapper._image_token_id = IMAGE_TOKEN_ID
    return wrapper


class TestK3ImagePromptIds(CustomTestCase):
    def test_expand_matches_reference_loop(self):
        tok = _StubTokenizer()
        for expand in (
            _expand_k3_image_prompt_token_ids,
            _expand_k3_image_prompt_token_ids_vectorized,
        ):
            for prompt in PROMPTS:
                with self.subTest(expand=expand.__name__, prompt=prompt):
                    counts, sizes = _case(prompt)
                    want = _expand_reference(prompt, IMAGE_TOKEN_ID, counts, sizes, tok)

                    got = expand(prompt, IMAGE_TOKEN_ID, counts, sizes, tok)
                    self.assertEqual(got.shape[0], 1)
                    self.assertEqual(got.dtype, torch.long)
                    self.assertEqual(got.flatten().tolist(), want)

                    got_tensor = expand(
                        torch.tensor(prompt), IMAGE_TOKEN_ID, counts, sizes, tok
                    )
                    self.assertEqual(got_tensor.flatten().tolist(), want)

    def test_expand_rejects_placeholder_count_mismatch(self):
        for expand in (
            _expand_k3_image_prompt_token_ids,
            _expand_k3_image_prompt_token_ids_vectorized,
        ):
            with self.subTest(expand=expand.__name__):
                with self.assertRaisesRegex(ValueError, "placeholder"):
                    expand(
                        [1, IMAGE_TOKEN_ID],
                        IMAGE_TOKEN_ID,
                        [2, 2],
                        [(8, 8), (8, 8)],
                        _StubTokenizer(),
                    )

    def _prepare_all(self, wrapper, prompts):
        outs = []
        for prompt in prompts:
            counts, sizes = _case(prompt)
            resize_configs = [{"num_tokens": c} for c in counts]
            outs.append(
                wrapper._prepare_input_ids(None, resize_configs, prompt, sizes)
                .flatten()
                .tolist()
            )
        return outs

    def test_flag_off_uses_loop_expansion(self):
        tok = _StubTokenizer()
        wrapper = _make_wrapper(tok)
        prompts = PROMPTS + [torch.tensor(p) for p in PROMPTS]

        self.assertIs(envs.SGLANG_K3_MM_USE_RENDERED_INPUT_IDS.get(), False)
        with patch.object(
            kimi_k3,
            "_expand_k3_image_prompt_token_ids",
            wraps=_expand_k3_image_prompt_token_ids,
        ) as loop, patch.object(
            kimi_k3,
            "_expand_k3_image_prompt_token_ids_vectorized",
            wraps=_expand_k3_image_prompt_token_ids_vectorized,
        ) as vectorized:
            got = self._prepare_all(wrapper, prompts)
        self.assertEqual(loop.call_count, len(prompts))
        self.assertEqual(vectorized.call_count, 0)

        want = [
            _expand_k3_image_prompt_token_ids_vectorized(
                p, IMAGE_TOKEN_ID, *_case(p), tok
            )
            .flatten()
            .tolist()
            for p in prompts
        ]
        self.assertEqual(got, want)

    def test_flag_on_uses_vectorized_expansion(self):
        tok = _StubTokenizer()
        wrapper = _make_wrapper(tok)
        prompts = PROMPTS + [torch.tensor(p) for p in PROMPTS]

        with envs.SGLANG_K3_MM_USE_RENDERED_INPUT_IDS.override(True), patch.object(
            kimi_k3,
            "_expand_k3_image_prompt_token_ids",
            wraps=_expand_k3_image_prompt_token_ids,
        ) as loop, patch.object(
            kimi_k3,
            "_expand_k3_image_prompt_token_ids_vectorized",
            wraps=_expand_k3_image_prompt_token_ids_vectorized,
        ) as vectorized:
            got = self._prepare_all(wrapper, prompts)
        self.assertEqual(vectorized.call_count, len(prompts))
        self.assertEqual(loop.call_count, 0)

        want = [
            _expand_k3_image_prompt_token_ids(p, IMAGE_TOKEN_ID, *_case(p), tok)
            .flatten()
            .tolist()
            for p in prompts
        ]
        self.assertEqual(got, want)

    def test_prepare_input_ids_keeps_renderer_ids_for_literal_special_text(self):
        """User text spelling ``<|sep|>`` stays plain text when the renderer's
        ids are passed through, but collapses to the control token on the
        retokenize path -- the documented difference between the two modes."""
        tok = _StubTokenizer()
        wrapper = _make_wrapper(tok)

        literal = tok.encode("<|sep|>")
        self.assertNotIn(SEP_CONTROL_ID, literal)
        rendered = literal + [IMAGE_TOKEN_ID]
        text = tok.decode(rendered)
        resize_configs = [{"num_tokens": 2}]
        sizes = [(16, 16)]

        passthrough = wrapper._prepare_input_ids(
            text, resize_configs, rendered, sizes
        ).flatten()
        retokenized = wrapper._prepare_input_ids(
            text, resize_configs, None, sizes
        ).flatten()

        self.assertEqual(passthrough.tolist()[: len(literal)], literal)
        self.assertEqual(retokenized.tolist()[0], SEP_CONTROL_ID)
        # Both agree on everything after the user text: the media framing.
        self.assertEqual(passthrough.tolist()[len(literal) :], retokenized.tolist()[1:])

    def _run(self, proc, prompt):
        async def go():
            with self.assertRaises(_Stop):
                await proc.process_mm_data_async(
                    image_data=[b"img"], input_text=prompt, request_obj=object()
                )

        asyncio.run(go())

    def test_renderer_id_passthrough_is_opt_in(self):
        prompt = [1, 2, IMAGE_TOKEN_ID]
        proc = _FastLoadCapture()

        self.assertIs(envs.SGLANG_K3_MM_USE_RENDERED_INPUT_IDS.get(), False)
        self._run(proc, prompt)
        self.assertIsNone(proc.captured_input_ids)

        with envs.SGLANG_K3_MM_USE_RENDERED_INPUT_IDS.override(True):
            self._run(proc, prompt)
        self.assertIs(proc.captured_input_ids, prompt)


if __name__ == "__main__":
    unittest.main()
