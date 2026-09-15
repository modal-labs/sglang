import copy
import itertools
import pickle
import random
import sys
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from unittest import mock

from transformers import AutoTokenizer

from sglang.srt.environ import envs
from sglang.srt.utils.patch_tokenizer import (
    _CHAT_SEGMENT_CACHE_ATTR,
    _ChatSegmentCachePatcher,
    _EncodePieceFastPathPatcher,
    _SpecialTokensCachePatcher,
    decode_without_hf_kwargs,
    patch_mm_processor_tokenizer,
    patch_tokenizer,
    unpatch_tokenizer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu", nightly=True)
register_cpu_ci(est_time=53, suite="base-c-test-cpu")


class TestPatchTokenizerEndToEndTest(unittest.TestCase):
    def test_patched_produces_same_results_as_raw(self):
        tokenizer = _load_tokenizer()
        test_texts = self._generate_test_texts(tokenizer)
        raw_results = self._run_tokenizer_ops(tokenizer, test_texts)

        _SpecialTokensCachePatcher.patch(tokenizer)
        patched_results = self._run_tokenizer_ops(tokenizer, test_texts)
        unpatch_tokenizer(tokenizer)

        self.assertEqual(raw_results, patched_results)

    @classmethod
    def _generate_test_texts(cls, tokenizer):
        special_tokens = tokenizer.all_special_tokens
        return [
            "Hello, world!",
            "This is a longer sentence with multiple words.",
            "Numbers 12345 and symbols !@#$%",
            "    leading and trailing spaces    ",
            "\n\nMultiple\n\nNewlines\n\n",
            *[f"Text with {tok} inside" for tok in special_tokens],
            " ".join(special_tokens),
            *[
                cls._random_text_from_tokens(tokenizer, num_tokens=100)
                for _ in range(5)
            ],
            *[
                cls._random_text_from_tokens(tokenizer, num_tokens=1000)
                for _ in range(3)
            ],
        ]

    @classmethod
    def _random_text_from_tokens(cls, tokenizer, num_tokens):
        token_ids = [
            random.randint(0, tokenizer.vocab_size - 1) for _ in range(num_tokens)
        ]
        return tokenizer.decode(token_ids)

    @classmethod
    def _run_tokenizer_ops(cls, tokenizer, texts):
        encode_results = [tokenizer.encode(t) for t in texts]
        batch_encode_results = tokenizer(texts)["input_ids"]
        return {
            "encode": encode_results,
            "batch_encode": batch_encode_results,
            "decode": [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in encode_results
            ],
            "batch_decode": tokenizer.batch_decode(
                encode_results, skip_special_tokens=True
            ),
            "special_tokens": tokenizer.all_special_tokens,
            "special_ids": tokenizer.all_special_ids,
        }


class TestPatchTokenizerUnitTest(unittest.TestCase):
    def test_patch_unpatch_restores_original(self):
        tokenizer = _load_tokenizer()
        cls = type(tokenizer)

        original_ids = _get_class_attr_ids(cls)

        _SpecialTokensCachePatcher.patch(tokenizer)
        self.assertTrue(getattr(cls, "_sglang_special_tokens_patched", False))

        patched_ids = _get_class_attr_ids(cls)
        changed_attrs = [
            name
            for name in original_ids
            if name in patched_ids and patched_ids[name] != original_ids[name]
        ]
        self.assertGreater(len(changed_attrs), 0, "Patch should change some attributes")

        unpatch_tokenizer(tokenizer)
        self.assertFalse(getattr(cls, "_sglang_special_tokens_patched", False))

        restored_ids = _get_class_attr_ids(cls)
        for name in original_ids:
            if name.startswith("_sglang") or name.startswith("_original"):
                continue
            self.assertEqual(
                restored_ids.get(name),
                original_ids[name],
                f"Attribute {name} should be restored to original",
            )

    def test_patch_caches_special_tokens(self):
        with _patched_tokenizer() as tokenizer:
            tokens1 = tokenizer.all_special_tokens
            ids1 = tokenizer.all_special_ids
            tokens2 = tokenizer.all_special_tokens
            ids2 = tokenizer.all_special_ids

            self.assertIs(tokens1, tokens2)
            self.assertIs(ids1, ids2)

    def test_patch_blocks_add_special_tokens(self):
        with _patched_tokenizer() as tokenizer:
            with self.assertRaises(AssertionError) as ctx:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.assertIn(
                "Cannot modify special tokens after patch", str(ctx.exception)
            )

    def test_patch_blocks_add_tokens_with_special_flag(self):
        with _patched_tokenizer() as tokenizer:
            with self.assertRaises(AssertionError) as ctx:
                tokenizer.add_tokens(["<new>"], special_tokens=True)
            self.assertIn("Cannot add special tokens after patch", str(ctx.exception))

            tokenizer.add_tokens(["<regular>"], special_tokens=False)

    def test_unpatch_clears_cache(self):
        with _patched_tokenizer() as tokenizer:
            _ = tokenizer.all_special_tokens
            _ = tokenizer.all_special_ids
            self.assertTrue(hasattr(tokenizer, "_sglang_cached_special_tokens"))
            self.assertTrue(hasattr(tokenizer, "_sglang_cached_special_ids"))

        self.assertFalse(hasattr(tokenizer, "_sglang_cached_special_tokens"))
        self.assertFalse(hasattr(tokenizer, "_sglang_cached_special_ids"))

    def test_double_patch_is_idempotent(self):
        tokenizer = _load_tokenizer()
        _SpecialTokensCachePatcher.patch(tokenizer)
        _SpecialTokensCachePatcher.patch(tokenizer)

        self.assertTrue(
            getattr(type(tokenizer), "_sglang_special_tokens_patched", False)
        )

        unpatch_tokenizer(tokenizer)

    def test_decode_without_hf_kwargs_uses_native_decode(self):
        tokenizer = _FakeDecodeTokenizer()

        self.assertEqual(
            decode_without_hf_kwargs(tokenizer, [1, 99, 2], True),
            "ab",
        )
        self.assertEqual(
            decode_without_hf_kwargs(tokenizer, [1, 99, 2], False),
            "a<special>b",
        )
        self.assertEqual(tokenizer.decode_calls, [[1, 2], [1, 99, 2]])


class TestEncodePieceFastPathPatcher(CustomTestCase):
    """The fast path must be a pure shortcut: every segment shape it handles
    (or declines) has to encode to the same ids as the original method."""

    def test_encode_piece_matches_original_on_every_segment_shape(self):
        tokenizer = _load_k3_tokenizer()
        specials = list(tokenizer.special_tokens)
        rng = random.Random(0)
        random_texts = [
            _random_text_from_tokens(tokenizer, num_tokens=n, rng=rng)
            for n in (1, 7, 100, 1000)
        ]
        segments = [
            # exactly one special token -> table lookup
            *[(tok, True) for tok in specials],
            # special token with a suffix / two specials -> original path
            *[(tok + "x", True) for tok in specials[:8]],
            (specials[0] + specials[1], True),
            # plain text -> encode_ordinary
            *[(text, False) for text in random_texts],
            ("", False),
            ("", True),
            (" " * 40, False),
            # special literal inside plain text -> original path
            *[(f"user wrote {tok} literally", False) for tok in specials[:8]],
            # lone surrogate (json.loads('"\\ud83d"') in a tool result) ->
            # tiktoken's UnicodeEncodeError fix-up must still apply
            ("tool output \ud83d broken", False),
            ("tool output \ud83d broken", True),
            # longer than MAX_NO_WHITESPACES_CHARS -> original splitter path
            ("b" * 30_000 + " tail", False),
            ("b" * 30_000 + " tail", True),
        ]
        original = type(tokenizer)._encode_text_piece
        _EncodePieceFastPathPatcher.patch(tokenizer)
        try:
            for text, allow_special in segments:
                self.assertEqual(
                    original(tokenizer, text, allow_special),
                    tokenizer._encode_text_piece(text, allow_special),
                    (text[:40], allow_special),
                )
        finally:
            _EncodePieceFastPathPatcher.unpatch(tokenizer)

    def test_chat_template_ids_unchanged_for_tool_call_conversation(self):
        # Kimi-K3's encoding_k3 renders one segment per control token or tag
        # name and four text segments per tool-call attribute; this is the
        # shape the fast path exists for.
        tokenizer = _load_k3_tokenizer()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                },
            }
        ]
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "deploy it"},
        ]
        for i in range(300):
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": "Bash",
                                "arguments": '{"command": "ls -la /tmp/%d"}' % i,
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "content": f"total 0\n<|im_end|> literal in tool output {i}",
                }
            )
        kwargs = dict(tools=tools, tokenize=True, add_generation_prompt=True)
        expected = tokenizer.apply_chat_template(messages, **kwargs)

        _EncodePieceFastPathPatcher.patch(tokenizer)
        try:
            self.assertEqual(
                expected, tokenizer.apply_chat_template(messages, **kwargs)
            )
        finally:
            _EncodePieceFastPathPatcher.unpatch(tokenizer)

    def test_unpatch_restores_encode_piece(self):
        tokenizer = _load_k3_tokenizer()
        cls = type(tokenizer)
        original = cls._encode_text_piece

        _EncodePieceFastPathPatcher.patch(tokenizer)
        self.assertIsNot(cls._encode_text_piece, original)
        _EncodePieceFastPathPatcher.patch(tokenizer)  # idempotent
        _EncodePieceFastPathPatcher.unpatch(tokenizer)

        self.assertIs(cls._encode_text_piece, original)
        self.assertFalse(hasattr(cls, "_original_encode_text_piece"))
        self.assertFalse(hasattr(tokenizer, "_sglang_special_literal_regex"))

    def test_k2_tokenizer_without_encode_text_piece_is_skipped(self):
        # The K2 family shares the TikTokenTokenizer class/module name but
        # inlines the segment loop into encode(); patch_tokenizer must still
        # apply the special-tokens cache and skip the encode-piece fast path.
        tokenizer = _load_tokenizer()
        cls = type(tokenizer)
        self.assertFalse(hasattr(cls, "_encode_text_piece"))
        self.assertFalse(_EncodePieceFastPathPatcher.applies_to(tokenizer))

        expected = tokenizer.encode("hello <|im_end|> world")
        with envs.SGLANG_PATCH_TOKENIZER.override(True):
            patched = patch_tokenizer(tokenizer)
        try:
            self.assertIs(patched, tokenizer)
            self.assertTrue(getattr(cls, "_sglang_special_tokens_patched", False))
            self.assertFalse(getattr(cls, "_sglang_encode_piece_patched", False))
            self.assertEqual(patched.encode("hello <|im_end|> world"), expected)
        finally:
            unpatch_tokenizer(tokenizer)


class TestComposedPatchers(CustomTestCase):
    """``patch_tokenizer`` stacks the chat-segment cache on top of the encode
    fast path: a cache miss must run the fast path, a hit must skip encoding
    entirely, and the ids must equal the unpatched tokenizer's."""

    def _messages(self):
        return [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "deploy it \ud83d please <|im_end|>"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "arguments": '{"command": "ls /tmp"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": "total 0"},
        ]

    def test_patch_unpatch_repatch_restores_class_exactly(self):
        tokenizer = _load_k3_tokenizer()
        cls = type(tokenizer)
        original = cls._encode_text_piece
        before = _get_class_attr_ids(cls)
        instance_attrs_before = set(vars(tokenizer))

        with envs.SGLANG_PATCH_TOKENIZER.override(
            True
        ), envs.SGLANG_KIMI_ENCODE_FAST_PATH.override(
            True
        ), envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(
            1_000_000
        ):
            for _ in range(2):
                patch_tokenizer(tokenizer)
                patch_tokenizer(tokenizer)  # idempotent
                self.assertTrue(getattr(cls, "_sglang_special_tokens_patched"))
                self.assertTrue(getattr(cls, "_sglang_encode_piece_patched"))
                self.assertTrue(getattr(cls, "_sglang_chat_segment_cache_patched"))
                # the cache's miss path is the fast path, whose fallback is the original
                self.assertEqual(
                    cls._sglang_original_encode_text_piece.__name__,
                    "patched_encode_text_piece",
                )
                self.assertIs(cls._original_encode_text_piece, original)
                tokenizer.apply_chat_template(self._messages(), tokenize=True)
                unpatch_tokenizer(tokenizer)
                self.assertEqual(_get_class_attr_ids(cls), before)
                self.assertEqual(set(vars(tokenizer)), instance_attrs_before)

    def test_miss_runs_fast_path_and_hit_skips_encoding(self):
        tokenizer = _load_k3_tokenizer()
        cls = type(tokenizer)
        original = cls._encode_text_piece
        messages = self._messages()
        expected = tokenizer.apply_chat_template(messages, tokenize=True)
        expected_text = tokenizer.encode("plain text segment " * 8)

        with envs.SGLANG_PATCH_TOKENIZER.override(
            True
        ), envs.SGLANG_KIMI_ENCODE_FAST_PATH.override(
            True
        ), envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(
            1_000_000
        ):
            patch_tokenizer(tokenizer)
            try:
                with mock.patch.object(
                    tokenizer.model,
                    "encode_ordinary",
                    wraps=tokenizer.model.encode_ordinary,
                ) as fast, mock.patch.object(
                    tokenizer.model, "encode", wraps=tokenizer.model.encode
                ) as slow:
                    self.assertEqual(
                        tokenizer.apply_chat_template(messages, tokenize=True), expected
                    )
                    self.assertEqual(
                        tokenizer.encode("plain text segment " * 8), expected_text
                    )
                    # misses: plain text went through encode_ordinary; only the
                    # segment with a special-token literal fell back to the original
                    self.assertGreater(fast.call_count, 0)
                    self.assertEqual(slow.call_count, 1)
                    state = getattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR)
                    misses = state.stats["misses"]
                    self.assertGreater(misses, 0)

                    fast.reset_mock()
                    slow.reset_mock()
                    self.assertEqual(
                        tokenizer.apply_chat_template(messages, tokenize=True), expected
                    )
                    self.assertEqual(
                        tokenizer.encode("plain text segment " * 8), expected_text
                    )
                    # hits: nothing was encoded again
                    self.assertEqual(fast.call_count, 0)
                    self.assertEqual(slow.call_count, 0)
                    self.assertEqual(state.stats["misses"], misses)
                    self.assertGreater(state.stats["hits"], 0)
            finally:
                unpatch_tokenizer(tokenizer)
        self.assertIs(cls._encode_text_piece, original)

    def test_cache_kill_switch_keeps_fast_path(self):
        tokenizer = _load_k3_tokenizer()
        cls = type(tokenizer)
        with envs.SGLANG_PATCH_TOKENIZER.override(
            True
        ), envs.SGLANG_KIMI_ENCODE_FAST_PATH.override(
            True
        ), envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(
            0
        ):
            patch_tokenizer(tokenizer)
        try:
            self.assertTrue(getattr(cls, "_sglang_encode_piece_patched"))
            self.assertFalse(hasattr(cls, "_sglang_chat_segment_cache_patched"))
            self.assertEqual(
                cls._encode_text_piece.__name__, "patched_encode_text_piece"
            )
        finally:
            unpatch_tokenizer(tokenizer)

    def test_defaults_apply_only_special_tokens_patch(self):
        # With nothing set the encode path is the model file's original.
        self.assertFalse(envs.SGLANG_KIMI_ENCODE_FAST_PATH.get())
        self.assertEqual(envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.get(), 0)
        self.assertTrue(envs.SGLANG_PATCH_TOKENIZER.get())
        tokenizer = _load_k3_tokenizer()
        cls = type(tokenizer)
        original = cls._encode_text_piece
        patch_tokenizer(tokenizer)
        try:
            self.assertTrue(getattr(cls, "_sglang_special_tokens_patched"))
            self.assertFalse(hasattr(cls, "_sglang_encode_piece_patched"))
            self.assertFalse(hasattr(cls, "_sglang_chat_segment_cache_patched"))
            self.assertIs(cls._encode_text_piece, original)
            self.assertFalse(hasattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR))
        finally:
            unpatch_tokenizer(tokenizer)
        self.assertIs(cls._encode_text_piece, original)

    def test_all_flag_combinations(self):
        tokenizer = _load_k3_tokenizer()
        cls = type(tokenizer)
        original = cls._encode_text_piece
        # The pre-existing special-tokens patcher's unpatch pins the restored
        # properties onto the subclass, so take the baseline after one cycle.
        unpatch_tokenizer(patch_tokenizer(tokenizer))
        before = _get_class_attr_ids(cls)
        instance_attrs_before = set(vars(tokenizer))
        messages = self._messages()
        expected = tokenizer.apply_chat_template(messages, tokenize=True)
        expected_text = tokenizer.encode("plain text segment <|im_end|> " * 8)

        for fast_path, max_chars in itertools.product((False, True), (0, 1_000_000)):
            with self.subTest(fast_path=fast_path, max_chars=max_chars):
                with envs.SGLANG_PATCH_TOKENIZER.override(
                    True
                ), envs.SGLANG_KIMI_ENCODE_FAST_PATH.override(
                    fast_path
                ), envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(
                    max_chars
                ):
                    patch_tokenizer(tokenizer)
                    patch_tokenizer(tokenizer)  # idempotent
                try:
                    self.assertTrue(getattr(cls, "_sglang_special_tokens_patched"))
                    self.assertEqual(
                        getattr(cls, "_sglang_encode_piece_patched", False),
                        fast_path,
                    )
                    self.assertEqual(
                        getattr(cls, "_sglang_chat_segment_cache_patched", False),
                        max_chars > 0,
                    )
                    # patch order: cache wraps fast path wraps original
                    if max_chars > 0:
                        self.assertEqual(
                            cls._encode_text_piece.__name__,
                            "cached_encode_text_piece",
                        )
                        inner = cls._sglang_original_encode_text_piece
                    else:
                        inner = cls._encode_text_piece
                    if fast_path:
                        self.assertEqual(inner.__name__, "patched_encode_text_piece")
                        self.assertIs(cls._original_encode_text_piece, original)
                    else:
                        self.assertIs(inner, original)
                    for _ in range(2):  # second pass exercises cache hits
                        self.assertEqual(
                            tokenizer.apply_chat_template(messages, tokenize=True),
                            expected,
                        )
                        self.assertEqual(
                            tokenizer.encode("plain text segment <|im_end|> " * 8),
                            expected_text,
                        )
                finally:
                    unpatch_tokenizer(tokenizer)
                self.assertIs(cls._encode_text_piece, original)
                self.assertEqual(_get_class_attr_ids(cls), before)
                self.assertEqual(set(vars(tokenizer)), instance_attrs_before)

    def test_differential_identity_fuzz(self):
        raw = _load_k3_tokenizer()
        special_tokens = [
            token for token in raw.special_tokens if token.startswith("<|")
        ]
        im_end = next(token for token in special_tokens if "end" in token)
        im_user = next(token for token in special_tokens if token != im_end)
        messages = self._messages()

        rng = random.Random(0)
        alphabet = (
            "abcXYZ0123 \t\n.,!?/\\"
            "你好世界こんにちは안녕"
            "😀🦙🚀"
            "<|im_end|><|im_user|>"
        )
        cases = [
            (
                f"random-{i}",
                "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 300))),
            )
            for i in range(200)
        ]
        cases.extend(
            [
                ("spaces-25000", " " * 25_000),
                ("spaces-25001", " " * 25_001),
                ("x-25000", "x" * 25_000),
                ("x-25001", "x" * 25_001),
                (f"{im_end}-alone", im_end),
                (f"{im_user}-alone", im_user),
                ("embedded", f"a{im_end}b"),
                ("adjacent", f"{im_end}{im_user}"),
                ("partial-left", im_end[:5]),
                ("partial-right", im_end[5:]),
                ("lone-surrogate-left", "\ud800"),
                ("lone-surrogate-middle", "a\udfffb"),
                ("tool-call-json", '{"name":"Bash","arguments":{"command":"ls /tmp"}}'),
            ]
        )

        expected = {
            (name, allow_special): raw._encode_text_piece(
                text, allow_special_tokens=allow_special
            )
            for name, text in cases
            for allow_special in (False, True)
        }
        expected_render = raw.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )

        def run_piece_cases(tokenizer):
            outputs = {}
            for name, text in cases:
                for allow_special in (False, True):
                    outputs[(name, allow_special)] = tokenizer._encode_text_piece(
                        text, allow_special_tokens=allow_special
                    )
            render = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            return outputs, render

        with self.subTest(mode="cache+fast"):
            with envs.SGLANG_PATCH_TOKENIZER.override(
                True
            ), envs.SGLANG_KIMI_ENCODE_FAST_PATH.override(
                True
            ), envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(
                1_000_000
            ):
                cached = patch_tokenizer(_load_k3_tokenizer())
                cached_outputs = None
                for _ in range(2):
                    cached_outputs, cached_render = run_piece_cases(cached)
                self.assertEqual(cached_outputs, expected)
                self.assertEqual(cached_render, expected_render)
                unpatch_tokenizer(cached)

        with self.subTest(mode="fast-only"):
            with envs.SGLANG_PATCH_TOKENIZER.override(
                True
            ), envs.SGLANG_KIMI_ENCODE_FAST_PATH.override(
                True
            ), envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(
                0
            ):
                fast_only = patch_tokenizer(_load_k3_tokenizer())
            try:
                fast_outputs = None
                for _ in range(2):
                    fast_outputs, fast_render = run_piece_cases(fast_only)
                self.assertEqual(fast_outputs, expected)
                self.assertEqual(fast_render, expected_render)
            finally:
                unpatch_tokenizer(fast_only)

        self.assertEqual(cached_outputs, fast_outputs)
        self.assertEqual(cached_render, fast_render)


def _random_text_from_tokens(tokenizer, num_tokens, rng):
    token_ids = [rng.randint(0, tokenizer.vocab_size - 1) for _ in range(num_tokens)]
    return tokenizer.decode(token_ids)


class TestChatSegmentCachePatcher(unittest.TestCase):
    """CPU-only coverage for ``_ChatSegmentCachePatcher`` on a Kimi-shaped fake.

    The real K3 tokenizer is gated on the Hub and the K2 tokenizer in CI has
    no ``_encode_chat_segments``, so the fake mirrors that interface.
    """

    def setUp(self):
        self.tokenizer_cls = _make_fake_kimi_tokenizer_cls()
        self.original_text_piece = self.tokenizer_cls._encode_text_piece
        self.original_segments = self.tokenizer_cls._encode_chat_segments
        self.messages = [
            _Segment("<|im_user|>", True),
            _Segment("hello world, this is a long enough user turn " * 4, False),
            _Segment("<|im_end|>", True),
            _Segment("<|im_assistant|>", True),
            _Segment("and a long enough assistant reply " * 4, False),
            _Segment("<|im_end|>", True),
        ]

    def _patched(self, max_chars=1_000_000):
        return _patched_chat_segment_tokenizer(self.tokenizer_cls, max_chars)

    def test_cached_ids_match_uncached(self):
        raw = self.tokenizer_cls()
        expected_segments = raw._encode_chat_segments(self.messages)
        expected_piece = raw._encode_text_piece(
            "plain text", allow_special_tokens=False
        )
        expected_piece_special = raw._encode_text_piece("<|im_end|>")

        with self._patched() as tokenizer:
            for _ in range(3):
                self.assertEqual(
                    tokenizer._encode_chat_segments(self.messages), expected_segments
                )
                self.assertEqual(
                    tokenizer._encode_text_piece(
                        "plain text", allow_special_tokens=False
                    ),
                    expected_piece,
                )
                self.assertEqual(
                    tokenizer._encode_text_piece("<|im_end|>"), expected_piece_special
                )

    def test_allow_special_is_part_of_the_key(self):
        with self._patched() as tokenizer:
            special = tokenizer._encode_text_piece(
                "<|im_end|>", allow_special_tokens=True
            )
            literal = tokenizer._encode_text_piece(
                "<|im_end|>", allow_special_tokens=False
            )
            self.assertNotEqual(special, literal)
            state = getattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR)
            self.assertEqual(len(state.cache), 2)

    def test_hits_and_misses_are_counted(self):
        with self._patched() as tokenizer:
            tokenizer._encode_chat_segments(self.messages)
            state = getattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR)
            self.assertEqual(state.stats["misses"], 5)  # <|im_end|> is a dup
            self.assertEqual(state.stats["hits"], 1)
            self.assertEqual(tokenizer.encode_calls, 5)

            tokenizer._encode_chat_segments(self.messages)
            self.assertEqual(state.stats["misses"], 5)
            self.assertEqual(state.stats["hits"], 1 + len(self.messages))
            self.assertEqual(tokenizer.encode_calls, 5)

    def test_eviction_stays_under_budget(self):
        budget = 300
        with self._patched(max_chars=budget) as tokenizer:
            for i in range(50):
                tokenizer._encode_text_piece(f"segment number {i} " * 5)
            state = getattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR)
            chars = sum(len(text) for (_, text) in state.cache)
            self.assertEqual(state.stats["chars"], chars)
            self.assertLessEqual(chars, budget)
            self.assertGreater(len(state.cache), 0)

    def test_oversize_segment_bypasses_cache(self):
        with self._patched(max_chars=10) as tokenizer:
            text = "x" * 100
            expected = self.original_text_piece(tokenizer, text)
            self.assertEqual(tokenizer._encode_text_piece(text), expected)
            self.assertEqual(tokenizer._encode_text_piece(text), expected)
            state = getattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR)
            self.assertEqual(len(state.cache), 0)
            self.assertEqual(tokenizer.encode_calls, 3)

    def test_lru_touch_keeps_recent_entries(self):
        with self._patched(max_chars=250) as tokenizer:
            a = "a" * 100
            b = "b" * 100
            tokenizer._encode_text_piece(a)
            tokenizer._encode_text_piece(b)
            tokenizer._encode_text_piece(a)  # touch a -> b is now oldest
            tokenizer._encode_text_piece("c" * 100)  # evicts b
            state = getattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR)
            self.assertIn((True, a), state.cache)
            self.assertNotIn((True, b), state.cache)

    def test_cache_is_per_instance(self):
        with self._patched() as tokenizer:
            other = self.tokenizer_cls()
            tokenizer._encode_chat_segments(self.messages)
            other._encode_chat_segments(self.messages)
            tokenizer_state = getattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR)
            other_state = getattr(other, _CHAT_SEGMENT_CACHE_ATTR)
            self.assertIsNot(tokenizer_state, other_state)
            self.assertEqual(tokenizer.encode_calls, other.encode_calls)
            self.assertEqual(
                tokenizer_state.stats["misses"], other_state.stats["misses"]
            )
            self.assertEqual(tokenizer_state.stats["hits"], other_state.stats["hits"])
            self.assertNotIn(tokenizer, tokenizer_state.cache)
            self.assertNotIn(other, tokenizer_state.cache)
            self.assertNotIn(tokenizer, other_state.cache)
            self.assertNotIn(other, other_state.cache)

    def test_add_tokens_invalidates_cache(self):
        raw = self.tokenizer_cls()
        original_add_tokens = self.tokenizer_cls.add_tokens
        with self._patched() as tokenizer:
            before = tokenizer._encode_text_piece(
                "custom marker", allow_special_tokens=False
            )
            state = getattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR)
            self.assertEqual(len(state.cache), 1)

            self.assertEqual(tokenizer.add_tokens(["custom marker"]), 1)
            raw.add_tokens(["custom marker"])
            self.assertEqual(len(state.cache), 0)
            self.assertEqual(state.stats["chars"], 0)

            after = tokenizer._encode_text_piece(
                "custom marker", allow_special_tokens=False
            )
            self.assertNotEqual(after, before)
            self.assertEqual(
                after,
                raw._encode_text_piece("custom marker", allow_special_tokens=False),
            )
        self.assertIs(self.tokenizer_cls.add_tokens, original_add_tokens)

    def test_mm_processor_tokenizer_is_untouched_by_default(self):
        tokenizer = self.tokenizer_cls()
        class_attrs = _get_class_attr_ids(self.tokenizer_cls)
        with envs.SGLANG_KIMI_ENCODE_FAST_PATH.override(
            False
        ), envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(0):
            self.assertIs(patch_mm_processor_tokenizer(tokenizer), tokenizer)
        self.assertEqual(_get_class_attr_ids(self.tokenizer_cls), class_attrs)
        self.assertFalse(hasattr(self.tokenizer_cls, "_sglang_special_tokens_patched"))

        with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(1_000_000):
            patch_mm_processor_tokenizer(tokenizer)
        try:
            self.assertTrue(
                getattr(self.tokenizer_cls, "_sglang_special_tokens_patched")
            )
            self.assertTrue(
                getattr(self.tokenizer_cls, "_sglang_chat_segment_cache_patched")
            )
        finally:
            unpatch_tokenizer(tokenizer)

    def test_kill_switch_leaves_class_unpatched(self):
        tokenizer = self.tokenizer_cls()
        with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(0):
            _ChatSegmentCachePatcher.patch(tokenizer)
        self.assertIs(self.tokenizer_cls._encode_text_piece, self.original_text_piece)
        self.assertIs(self.tokenizer_cls._encode_chat_segments, self.original_segments)
        self.assertFalse(
            hasattr(self.tokenizer_cls, "_sglang_chat_segment_cache_patched")
        )

    def test_small_cap_warning_threshold(self):
        tokenizer = self.tokenizer_cls()
        with self.assertLogs(
            "sglang.srt.utils.patch_tokenizer", level="WARNING"
        ) as logs:
            with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(100_000):
                patch_tokenizer(tokenizer)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("32000000", logs.output[0])

        other = self.tokenizer_cls()
        with self.assertNoLogs("sglang.srt.utils.patch_tokenizer", level="WARNING"):
            with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(100_000):
                patch_tokenizer(other)
        unpatch_tokenizer(other)
        unpatch_tokenizer(tokenizer)

        tokenizer = self.tokenizer_cls()
        with self.assertNoLogs("sglang.srt.utils.patch_tokenizer", level="WARNING"):
            with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(32_000_000):
                patch_tokenizer(tokenizer)
                tokenizer._encode_chat_segments(self.messages)
        unpatch_tokenizer(tokenizer)

        tokenizer = self.tokenizer_cls()
        with self.assertNoLogs("sglang.srt.utils.patch_tokenizer", level="WARNING"):
            with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(0):
                patch_tokenizer(tokenizer)
                tokenizer._encode_chat_segments(self.messages)
        self.assertFalse(
            hasattr(self.tokenizer_cls, "_sglang_chat_segment_cache_patched")
        )
        self.assertFalse(
            hasattr(self.tokenizer_cls, "_sglang_original_encode_text_piece")
        )
        self.assertNotIn(_CHAT_SEGMENT_CACHE_ATTR, tokenizer.__dict__)
        unpatch_tokenizer(tokenizer)

    def test_unpatch_restores_originals_and_clears_cache(self):
        tokenizer = self.tokenizer_cls()
        with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(1_000_000):
            _ChatSegmentCachePatcher.patch(tokenizer)
        self.assertIsNot(
            self.tokenizer_cls._encode_text_piece, self.original_text_piece
        )
        tokenizer._encode_chat_segments(self.messages)
        self.assertTrue(hasattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR))

        unpatch_tokenizer(tokenizer)
        self.assertIs(self.tokenizer_cls._encode_text_piece, self.original_text_piece)
        self.assertIs(self.tokenizer_cls._encode_chat_segments, self.original_segments)
        self.assertFalse(
            hasattr(self.tokenizer_cls, "_sglang_chat_segment_cache_patched")
        )
        self.assertFalse(
            hasattr(self.tokenizer_cls, "_sglang_original_encode_text_piece")
        )
        self.assertFalse(hasattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR))
        tokenizer._encode_chat_segments(self.messages)
        self.assertNotIn(_CHAT_SEGMENT_CACHE_ATTR, tokenizer.__dict__)

        # re-patch works after unpatch
        with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(1_000_000):
            _ChatSegmentCachePatcher.patch(tokenizer)
        self.assertIsNot(
            self.tokenizer_cls._encode_text_piece, self.original_text_piece
        )
        unpatch_tokenizer(tokenizer)
        self.assertIs(self.tokenizer_cls._encode_text_piece, self.original_text_piece)

        original_ids = _get_class_attr_ids(self.tokenizer_cls)
        with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(1_000_000):
            patch_tokenizer(tokenizer)
        unpatch_tokenizer(tokenizer)
        self.assertEqual(_get_class_attr_ids(self.tokenizer_cls), original_ids)

    def test_unpatch_tokenizer_undoes_both_patchers(self):
        tokenizer = self.tokenizer_cls()
        with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(1_000_000):
            patch_tokenizer(tokenizer)
        self.assertTrue(getattr(self.tokenizer_cls, "_sglang_special_tokens_patched"))
        self.assertTrue(
            getattr(self.tokenizer_cls, "_sglang_chat_segment_cache_patched")
        )

        unpatch_tokenizer(tokenizer)
        self.assertFalse(hasattr(self.tokenizer_cls, "_sglang_special_tokens_patched"))
        self.assertFalse(
            hasattr(self.tokenizer_cls, "_sglang_chat_segment_cache_patched")
        )
        self.assertIs(self.tokenizer_cls._encode_text_piece, self.original_text_piece)

    def test_unpatch_via_one_instance_invalidates_other_instances_caches(self):
        tokenizer = self.tokenizer_cls()
        other = self.tokenizer_cls()
        text = "hello world"
        with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(1_000_000):
            patch_tokenizer(tokenizer)
            patch_tokenizer(other)
            original_ids = tokenizer._encode_text_piece(
                text, allow_special_tokens=False
            )
            other._encode_text_piece(text, allow_special_tokens=False)
        old_other_state = getattr(other, _CHAT_SEGMENT_CACHE_ATTR)

        unpatch_tokenizer(tokenizer)
        try:
            other.add_tokens([text])
            expected_new_ids = other._encode_text_piece(
                text, allow_special_tokens=False
            )
            with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(1_000_000):
                patch_tokenizer(other)
                patch_tokenizer(tokenizer)
            self.assertNotEqual(expected_new_ids, original_ids)

            new_other_ids = other._encode_text_piece(text, allow_special_tokens=False)
            new_other_state = getattr(other, _CHAT_SEGMENT_CACHE_ATTR)
            self.assertEqual(new_other_ids, expected_new_ids)
            self.assertIsNot(new_other_state, old_other_state)
            self.assertEqual(new_other_state.stats["misses"], 1)
            self.assertEqual(new_other_state.stats["hits"], 0)

            new_tokenizer_ids = tokenizer._encode_text_piece(
                text, allow_special_tokens=False
            )
            new_tokenizer_state = getattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR)
            self.assertEqual(new_tokenizer_ids, original_ids)
            self.assertEqual(new_tokenizer_state.stats["misses"], 1)
            self.assertEqual(new_tokenizer_state.stats["hits"], 0)
        finally:
            unpatch_tokenizer(other)
            unpatch_tokenizer(tokenizer)

    def test_unpatch_drops_instance_cache_even_when_class_already_unpatched(self):
        tokenizer = self.tokenizer_cls()
        other = self.tokenizer_cls()
        with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(1_000_000):
            patch_tokenizer(tokenizer)
            patch_tokenizer(other)
            tokenizer._encode_text_piece("hello world")
            other._encode_text_piece("hello world")

        unpatch_tokenizer(tokenizer)
        self.assertIn(_CHAT_SEGMENT_CACHE_ATTR, other.__dict__)
        unpatch_tokenizer(other)
        self.assertNotIn(_CHAT_SEGMENT_CACHE_ATTR, other.__dict__)
        self.assertFalse(
            any(name.startswith("_sglang_") for name in vars(self.tokenizer_cls))
        )

    def test_deep_copies_are_independent(self):
        with self._patched() as tokenizer:
            tokenizer._encode_chat_segments(self.messages)
            state = getattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR)
            misses = state.stats["misses"]

            clone = copy.deepcopy(tokenizer)
            self.assertIsNone(clone.__dict__.get(_CHAT_SEGMENT_CACHE_ATTR))
            clone._encode_chat_segments(self.messages)
            clone_state = getattr(clone, _CHAT_SEGMENT_CACHE_ATTR)
            self.assertIsNot(clone_state, state)
            self.assertEqual(clone_state.stats["misses"], 5)
            self.assertEqual(state.stats["misses"], misses)

            clone.add_tokens(["custom marker"])
            self.assertEqual(len(state.cache), 5)
            self.assertEqual(
                tokenizer._encode_text_piece(
                    "custom marker", allow_special_tokens=False
                ),
                [ord(c) % 997 for c in "custom marker"],
            )
            self.assertEqual(
                clone._encode_text_piece("custom marker", allow_special_tokens=False),
                [clone.added_tokens["custom marker"]],
            )

            shallow = copy.copy(tokenizer)
            misses_before_shallow = state.stats["misses"]
            shallow._encode_text_piece("copy")
            self.assertIsNot(getattr(shallow, _CHAT_SEGMENT_CACHE_ATTR), state)
            self.assertEqual(state.stats["misses"], misses_before_shallow)

            restored = pickle.loads(pickle.dumps(tokenizer))
            self.assertIsNone(restored.__dict__.get(_CHAT_SEGMENT_CACHE_ATTR))
            restored._encode_text_piece("pickle")
            self.assertIsNot(getattr(restored, _CHAT_SEGMENT_CACHE_ATTR), state)

    def test_add_tokens_clears_only_the_mutated_instance(self):
        with self._patched() as tokenizer:
            other = self.tokenizer_cls()
            text = "custom marker"
            original = tokenizer._encode_text_piece(text, allow_special_tokens=False)
            other_original = other._encode_text_piece(text, allow_special_tokens=False)
            tokenizer_state = getattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR)
            other_state = getattr(other, _CHAT_SEGMENT_CACHE_ATTR)

            other.add_tokens([text])
            self.assertIn((False, text), tokenizer_state.cache)
            self.assertEqual(
                tokenizer._encode_text_piece(text, allow_special_tokens=False),
                original,
            )
            self.assertEqual(len(other_state.cache), 0)
            self.assertNotEqual(
                other._encode_text_piece(text, allow_special_tokens=False),
                other_original,
            )

            untouched = self.tokenizer_cls()
            self.assertNotIn(_CHAT_SEGMENT_CACHE_ATTR, untouched.__dict__)
            untouched.add_tokens(["new token"])
            self.assertNotIn(_CHAT_SEGMENT_CACHE_ATTR, untouched.__dict__)

    def test_capacity_is_read_per_instance(self):
        with self._patched(max_chars=1_000_000) as tokenizer:
            with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(250):
                tokenizer._encode_text_piece("a" * 100)
            other = self.tokenizer_cls()
            with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(1_000):
                _ChatSegmentCachePatcher.patch(other)
                other._encode_text_piece("a" * 100)
            state = getattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR)
            other_state = getattr(other, _CHAT_SEGMENT_CACHE_ATTR)
            self.assertEqual(state.max_chars, 250)
            self.assertEqual(other_state.max_chars, 1_000)
            for text in ("b" * 100, "c" * 100):
                tokenizer._encode_text_piece(text)
                other._encode_text_piece(text)
            self.assertNotIn((True, "a" * 100), state.cache)
            self.assertIn((True, "a" * 100), other_state.cache)
            self.assertEqual(len(other_state.cache), 3)
            unpatch_tokenizer(other)

        disabled = self.tokenizer_cls()
        with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(1_000_000):
            _ChatSegmentCachePatcher.patch(disabled)
        with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(0):
            self.assertEqual(
                disabled._encode_text_piece("disabled"),
                [ord(c) % 997 for c in "disabled"],
            )
        self.assertEqual(len(getattr(disabled, _CHAT_SEGMENT_CACHE_ATTR).cache), 0)
        unpatch_tokenizer(disabled)

    def test_model_replacement_resets_cache(self):
        with self._patched() as tokenizer:
            expected = tokenizer._encode_text_piece("model")
            state = getattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR)
            tokenizer.model = object()
            self.assertEqual(tokenizer._encode_text_piece("model"), expected)
            new_state = getattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR)
            self.assertIsNot(new_state, state)
            self.assertEqual(new_state.stats["misses"], 1)

    def test_deepcopy_and_pickle_after_encode(self):
        with self._patched() as tokenizer:
            expected = tokenizer._encode_chat_segments(self.messages)
            clone = copy.deepcopy(tokenizer)
            self.assertIsNone(clone.__dict__.get(_CHAT_SEGMENT_CACHE_ATTR))
            self.assertEqual(clone._encode_chat_segments(self.messages), expected)

            restored = pickle.loads(pickle.dumps(tokenizer))
            self.assertIsNone(restored.__dict__.get(_CHAT_SEGMENT_CACHE_ATTR))
            self.assertEqual(restored._encode_chat_segments(self.messages), expected)

    def test_shallow_copy_gets_its_own_cache(self):
        with self._patched() as tokenizer:
            tokenizer._encode_text_piece("copy")
            state = getattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR)
            misses = state.stats["misses"]

            clone = copy.copy(tokenizer)
            self.assertIs(getattr(clone, _CHAT_SEGMENT_CACHE_ATTR), state)
            clone._encode_text_piece("copy")
            clone_state = getattr(clone, _CHAT_SEGMENT_CACHE_ATTR)
            self.assertIsNot(clone_state, state)
            self.assertEqual(state.stats["misses"], misses)
            self.assertEqual(clone_state.stats["misses"], 1)

    def test_flags_off_encode_adds_no_instance_state(self):
        tokenizer = self.tokenizer_cls()
        with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(0):
            patch_tokenizer(tokenizer)
            tokenizer._encode_chat_segments(self.messages)
        self.assertNotIn(_CHAT_SEGMENT_CACHE_ATTR, tokenizer.__dict__)
        self.assertFalse(
            hasattr(self.tokenizer_cls, "_sglang_original_encode_text_piece")
        )
        unpatch_tokenizer(tokenizer)

    def test_special_tokens_not_cached_as_literal_and_blocked(self):
        tokenizer = self.tokenizer_cls()
        with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(1_000_000):
            patch_tokenizer(tokenizer)
            try:
                special = tokenizer._encode_text_piece("<|im_end|>", True)
                literal = tokenizer._encode_text_piece("<|im_end|>", False)
                self.assertEqual(special, [1002])
                self.assertNotEqual(literal, [1002])
                state = getattr(tokenizer, _CHAT_SEGMENT_CACHE_ATTR)
                with self.assertRaises(AssertionError):
                    tokenizer.add_tokens(["<|new|>"], special_tokens=True)
                self.assertEqual(len(state.cache), 2)
            finally:
                unpatch_tokenizer(tokenizer)


@dataclass
class _Segment:
    text: str
    allow_special: bool


def _make_fake_kimi_tokenizer_cls():
    """A fresh ``TikTokenTokenizer`` lookalike per test so class-level patches never leak."""
    special_ids = {"<|im_user|>": 1000, "<|im_assistant|>": 1001, "<|im_end|>": 1002}

    class TikTokenTokenizer:
        name_or_path = "fake/kimi"
        vocab_size = 2000

        def __init__(self):
            self.encode_calls = 0
            self.added_tokens = {}
            self.model = object()

        @property
        def all_special_tokens(self):
            return list(special_ids)

        @property
        def all_special_ids(self):
            return list(special_ids.values())

        def add_special_tokens(self, *args, **kwargs):
            return 0

        def add_tokens(self, new_tokens, special_tokens=False):
            for token in new_tokens:
                self.added_tokens.setdefault(token, 1500 + len(self.added_tokens))
            return len(new_tokens)

        def _encode_text_piece(self, text, allow_special_tokens=True):
            self.encode_calls += 1
            if allow_special_tokens and text in special_ids:
                ids = [special_ids[text]]
            elif text in self.added_tokens:
                ids = [self.added_tokens[text]]
            else:
                ids = [ord(c) % 997 for c in text]
            return ids

        def _encode_chat_segments(self, segments):
            out = []
            for segment in segments:
                out.extend(
                    self._encode_text_piece(
                        segment.text, allow_special_tokens=segment.allow_special
                    )
                )
            return out

    # patch_tokenizer() detects Kimi by class name + module
    module = types.ModuleType("tokenization_kimi_fake")
    TikTokenTokenizer.__module__ = module.__name__
    TikTokenTokenizer.__qualname__ = TikTokenTokenizer.__name__
    module.TikTokenTokenizer = TikTokenTokenizer
    sys.modules[module.__name__] = module
    return TikTokenTokenizer


@contextmanager
def _patched_chat_segment_tokenizer(tokenizer_cls, max_chars):
    tokenizer = tokenizer_cls()
    with envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.override(max_chars):
        _ChatSegmentCachePatcher.patch(tokenizer)
        try:
            yield tokenizer
        finally:
            _ChatSegmentCachePatcher.unpatch(tokenizer)


def _get_class_attr_ids(cls):
    return {
        n: id(v.fget if isinstance(v, property) else v) for n, v in vars(cls).items()
    }


def _load_tokenizer():
    # The slowness is mainly observed in Kimi
    return AutoTokenizer.from_pretrained(
        "nvidia/Kimi-K2-Thinking-NVFP4", trust_remote_code=True
    )


def _load_k3_tokenizer():
    # Only Kimi-K3's tokenization_kimi.py has _encode_text_piece / encoding_k3.
    return AutoTokenizer.from_pretrained("moonshotai/Kimi-K3", trust_remote_code=True)


@contextmanager
def _patched_tokenizer():
    tokenizer = _load_tokenizer()
    _SpecialTokensCachePatcher.patch(tokenizer)
    try:
        yield tokenizer
    finally:
        unpatch_tokenizer(tokenizer)


class _FakeDecodeTokenizer:
    all_special_ids_set = {99}

    def __init__(self):
        self.decode_calls = []

    def decode(self, token_ids):
        token_ids = list(token_ids)
        self.decode_calls.append(token_ids)
        token_text = {1: "a", 2: "b", 99: "<special>"}
        return "".join(token_text[token_id] for token_id in token_ids)


if __name__ == "__main__":
    unittest.main()
