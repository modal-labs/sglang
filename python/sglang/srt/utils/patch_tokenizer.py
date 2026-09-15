import logging
import re
import weakref
from array import array
from collections import OrderedDict
from typing import List

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def patch_tokenizer(tokenizer):
    if not envs.SGLANG_PATCH_TOKENIZER.get():
        return tokenizer

    if _is_kimi_tiktoken_tokenizer(tokenizer):
        logger.info(
            f"Applying special tokens cache patch for Kimi tokenizer: {type(tokenizer)}"
        )
        tokenizer = _SpecialTokensCachePatcher.patch(tokenizer)
        if (
            envs.SGLANG_KIMI_ENCODE_FAST_PATH.get()
            and _EncodePieceFastPathPatcher.applies_to(tokenizer)
        ):
            logger.info(
                f"Applying encode-piece fast path patch for Kimi tokenizer: {type(tokenizer)}"
            )
            _EncodePieceFastPathPatcher.patch(tokenizer)
        _ChatSegmentCachePatcher.patch(tokenizer)

    return tokenizer


def patch_mm_processor_tokenizer(tokenizer):
    """Patch a multimodal processor's tokenizer only when an opt-in encode
    optimization is enabled; by default the processor tokenizer is left as
    loaded so the image re-tokenize path is unchanged."""
    if (
        envs.SGLANG_KIMI_ENCODE_FAST_PATH.get()
        or envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.get() > 0
    ):
        return patch_tokenizer(tokenizer)
    return tokenizer


def unpatch_tokenizer(tokenizer):
    tokenizer = _ChatSegmentCachePatcher.unpatch(tokenizer)
    tokenizer = _EncodePieceFastPathPatcher.unpatch(tokenizer)
    return _SpecialTokensCachePatcher.unpatch(tokenizer)


def _is_kimi_tiktoken_tokenizer(tokenizer):
    cls = type(tokenizer)
    class_name = cls.__name__
    module_name = cls.__module__ or ""
    return class_name == "TikTokenTokenizer" and "tokenization_kimi" in module_name


def decode_without_hf_kwargs(tokenizer, token_ids, skip_special_tokens):
    if skip_special_tokens:
        special_ids = getattr(tokenizer, "all_special_ids_set", None)
        if special_ids is None:
            special_ids = getattr(tokenizer, "all_special_ids", None)
        if special_ids is not None:
            special_ids_set = set(special_ids)
            token_ids = [tid for tid in token_ids if tid not in special_ids_set]
    return tokenizer.decode(token_ids)


_CHAT_SEGMENT_CACHE_ATTR = "_sglang_chat_segment_cache"


def _no_cache():
    return None


class _SegmentCache:
    """Per-instance memo for ``_encode_text_piece``.

    Owned by exactly one tokenizer instance and, like that instance, used from
    one thread at a time (the multimodal executor binds one deep-copied clone
    per worker thread; the tokenizer manager's tokenizer runs on the event
    loop), so there is no lock.  Copies of the owning tokenizer start with no
    cache: ``copy``/``deepcopy``/``pickle`` of this object yield ``None`` and
    the patcher creates a fresh cache on the clone's first encode.
    """

    __slots__ = ("cache", "stats", "max_chars", "model", "owner", "generation")

    def __init__(self, tokenizer, max_chars):
        self.cache = OrderedDict()
        self.stats = {"chars": 0, "hits": 0, "misses": 0}
        self.max_chars = max_chars
        self.model = tokenizer.model
        self.owner = weakref.ref(tokenizer)
        self.generation = _ChatSegmentCachePatcher._generation

    def __deepcopy__(self, memo):
        return None

    def __copy__(self):
        return None

    def __reduce__(self):
        return (_no_cache, ())


def _cache(self):
    st = self.__dict__.get(_CHAT_SEGMENT_CACHE_ATTR)
    if (
        st is None
        or st.owner() is not self
        or st.model is not self.model
        or st.generation != _ChatSegmentCachePatcher._generation
    ):
        st = _SegmentCache(self, envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.get())
        self.__dict__[_CHAT_SEGMENT_CACHE_ATTR] = st
    return st


class _SpecialTokensCachePatcher:
    _PATCHED_FLAG = "_sglang_special_tokens_patched"
    _CACHED_TOKENS_ATTR = "_sglang_cached_special_tokens"
    _CACHED_IDS_ATTR = "_sglang_cached_special_ids"

    @classmethod
    def patch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        tokenizer_cls._original_all_special_tokens = (
            tokenizer_cls.all_special_tokens.fget
        )
        tokenizer_cls._original_all_special_ids = tokenizer_cls.all_special_ids.fget
        tokenizer_cls._original_add_special_tokens = tokenizer_cls.add_special_tokens
        tokenizer_cls._original_add_tokens = tokenizer_cls.add_tokens

        patched_all_special_tokens = _make_cached_property(
            cls._CACHED_TOKENS_ATTR, tokenizer_cls._original_all_special_tokens
        )
        patched_all_special_ids = _make_cached_property(
            cls._CACHED_IDS_ATTR, tokenizer_cls._original_all_special_ids
        )

        def patched_add_special_tokens(self, *args, **kwargs):
            assert (
                False
            ), "Cannot modify special tokens after patch. Call unpatch_tokenizer first."

        def patched_add_tokens(self, new_tokens, special_tokens=False):
            assert (
                not special_tokens
            ), "Cannot add special tokens after patch. Call unpatch_tokenizer first."
            return tokenizer_cls._original_add_tokens(
                self, new_tokens, special_tokens=False
            )

        tokenizer_cls.all_special_tokens = patched_all_special_tokens
        tokenizer_cls.all_special_ids = patched_all_special_ids
        tokenizer_cls.add_special_tokens = patched_add_special_tokens
        tokenizer_cls.add_tokens = patched_add_tokens
        setattr(tokenizer_cls, cls._PATCHED_FLAG, True)

        return tokenizer

    @classmethod
    def unpatch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)
        if hasattr(tokenizer, _SPECIAL_LITERAL_REGEX_ATTR):
            delattr(tokenizer, _SPECIAL_LITERAL_REGEX_ATTR)

        if not getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        tokenizer_cls.all_special_tokens = property(
            tokenizer_cls._original_all_special_tokens
        )
        tokenizer_cls.all_special_ids = property(
            tokenizer_cls._original_all_special_ids
        )
        tokenizer_cls.add_special_tokens = tokenizer_cls._original_add_special_tokens
        tokenizer_cls.add_tokens = tokenizer_cls._original_add_tokens

        del tokenizer_cls._original_all_special_tokens
        del tokenizer_cls._original_all_special_ids
        del tokenizer_cls._original_add_special_tokens
        del tokenizer_cls._original_add_tokens
        delattr(tokenizer_cls, cls._PATCHED_FLAG)

        for attr in [cls._CACHED_TOKENS_ATTR, cls._CACHED_IDS_ATTR]:
            if hasattr(tokenizer, attr):
                delattr(tokenizer, attr)

        logger.info(f"Unpatched special tokens cache for {tokenizer_cls.__name__}")
        return tokenizer


class _EncodePieceFastPathPatcher:
    """Short-circuit ``TikTokenTokenizer._encode_text_piece`` for the two segment
    shapes that dominate Kimi-K3 chat encoding.

    ``encoding_k3.build_chat_segments`` renders a conversation into tens of
    thousands of tiny segments -- one per control token or tag name, and four
    text segments per tool-call attribute (`` key``, ``="``, value, ``"``) --
    and ``_encode_text_piece`` is called once per segment.  Every call first
    runs a pure-Python per-character splitter over the segment (control
    segments included); it yields the input unchanged below
    ``MAX_NO_WHITESPACES_CHARS`` but still costs O(len) Python per call.  Then:

    * control segments call ``tiktoken.Encoding.encode(allowed_special="all")``,
      which passes the cached 256-entry special-token set through the
      Python/Rust boundary on every call -- a fixed ~15-30us per call that
      dominates for tiny segments -- for what is a dictionary lookup;
    * text segments call ``encode(disallowed_special=())``, which for text
      with no special-token literal is exactly ``encode_ordinary``.

    The patched method keeps the original as the fallback, so token ids are
    unchanged: a special-token literal inside a text segment, a control segment
    that is not exactly one special token, and long text all take the original
    path.

    Only the Kimi-K3 tokenizer has ``_encode_text_piece``; the K2 family
    inlines the same loop into ``encode``, so ``applies_to`` must be checked
    before ``patch``.
    """

    _PATCHED_FLAG = "_sglang_encode_piece_patched"
    # Mirrors MAX_NO_WHITESPACES_CHARS in tokenization_kimi.py: below this length
    # the original splitter yields the input unchanged, so skipping it is exact.
    _MAX_UNSPLIT_TEXT_CHARS = 25_000

    @classmethod
    def applies_to(cls, tokenizer) -> bool:
        return callable(getattr(type(tokenizer), "_encode_text_piece", None))

    @classmethod
    def patch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer
        if not cls.applies_to(tokenizer):
            logger.info(
                f"Skipping encode-piece fast path: {tokenizer_cls.__name__} has no _encode_text_piece"
            )
            return tokenizer

        original_encode_text_piece = tokenizer_cls._encode_text_piece
        max_unsplit_text_chars = cls._MAX_UNSPLIT_TEXT_CHARS

        def patched_encode_text_piece(
            self, text: str, allow_special_tokens: bool = True
        ) -> List[int]:
            if allow_special_tokens:
                special_id = self.special_tokens.get(text)
                if special_id is not None:
                    return [special_id]
                return original_encode_text_piece(self, text, allow_special_tokens)
            if len(text) <= max_unsplit_text_chars and not _special_literal_regex(
                self
            ).search(text):
                # disallowed_special=() encodes special literals as plain text,
                # so with none present encode() == encode_ordinary().  Go through
                # the public method, not _core_bpe, to keep tiktoken's
                # UnicodeEncodeError fix-up for lone surrogates in the text.
                return self.model.encode_ordinary(text)
            return original_encode_text_piece(self, text, allow_special_tokens)

        tokenizer_cls._original_encode_text_piece = original_encode_text_piece
        tokenizer_cls._encode_text_piece = patched_encode_text_piece
        setattr(tokenizer_cls, cls._PATCHED_FLAG, True)
        return tokenizer

    @classmethod
    def unpatch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if not getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        tokenizer_cls._encode_text_piece = tokenizer_cls._original_encode_text_piece
        del tokenizer_cls._original_encode_text_piece
        delattr(tokenizer_cls, cls._PATCHED_FLAG)

        logger.info(f"Unpatched encode-piece fast path for {tokenizer_cls.__name__}")
        return tokenizer


_SPECIAL_LITERAL_REGEX_ATTR = "_sglang_special_literal_regex"


def _special_literal_regex(tokenizer):
    regex = getattr(tokenizer, _SPECIAL_LITERAL_REGEX_ATTR, None)
    if regex is None:
        regex = re.compile(
            "|".join(
                re.escape(token)
                for token in sorted(tokenizer.special_tokens, key=len, reverse=True)
            )
        )
        setattr(tokenizer, _SPECIAL_LITERAL_REGEX_ATTR, regex)
    return regex


class _ChatSegmentCachePatcher:
    """Memoize ``TikTokenTokenizer._encode_text_piece`` across requests.

    Kimi's chat encoder (``encoding_k3.build_chat_segments``) renders each
    message body as its own segment and encodes segments independently, so a
    per-(text, allow_special) memo reproduces the exact token ids while letting
    a multi-turn follow-up skip re-encoding its shared history.

    The memo is per tokenizer instance, with capacity read from
    ``SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS`` when that instance's cache is
    created. Copies start empty, ``add_tokens`` clears only that instance, and
    replacing ``self.model`` resets the cache. A cache found on an instance it
    does not own (such as a shallow copy) is discarded and rebuilt.
    Caches created before an ``unpatch`` are stamped with an older generation
    and rebuilt on the next patched encode.
    """

    _PATCHED_FLAG = "_sglang_chat_segment_cache_patched"
    _ORIGINAL_TEXT_PIECE_ATTR = "_sglang_original_encode_text_piece"
    _ORIGINAL_SEGMENTS_ATTR = "_sglang_original_encode_chat_segments"
    _ORIGINAL_ADD_TOKENS_ATTR = "_sglang_original_add_tokens_for_segment_cache"
    _SMALL_CAP_WARNING_CHARS = 1_000_000
    _warned_small_cap = False
    _generation = 0
    # segments shorter than this are structural markers (``<|sep|>``, ``="``,
    # ``message``); they are cached like everything else but never LRU-touched
    # since they are a few dozen distinct strings with negligible budget cost.
    _LRU_TOUCH_CHARS = 64

    @classmethod
    def patch(cls, tokenizer):
        max_chars = envs.SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS.get()
        tokenizer_cls = type(tokenizer)
        if max_chars <= 0:
            return tokenizer
        if max_chars < cls._SMALL_CAP_WARNING_CHARS and not cls._warned_small_cap:
            cls._warned_small_cap = True
            logger.warning(
                "SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS=%d is smaller than one typical "
                "long-context history (~400k chars per 100k tokens); the chat-segment "
                "cache will likely thrash and add overhead. Recommended: 32000000 "
                "(chars of message text, per tokenizer instance).",
                max_chars,
            )
        if getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer
        original = getattr(tokenizer_cls, "_encode_text_piece", None)
        original_segments = getattr(tokenizer_cls, "_encode_chat_segments", None)
        original_add_tokens = getattr(tokenizer_cls, "add_tokens", None)
        if original is None or original_segments is None or original_add_tokens is None:
            return tokenizer

        def lookup(self, key):
            st = _cache(self)
            hit = st.cache.get(key)
            if hit is not None and len(key[1]) >= cls._LRU_TOUCH_CHARS:
                st.cache.move_to_end(key)
            return hit

        def insert(self, key):
            st = _cache(self)
            text = key[1]
            ids = original(self, text, allow_special_tokens=key[0])
            st.stats["misses"] += 1
            n = len(text)
            if st.max_chars <= 0 or n > st.max_chars:
                return ids
            if key not in st.cache:
                st.cache[key] = array("i", ids)
                st.stats["chars"] += n
            while st.stats["chars"] > st.max_chars and st.cache:
                (_, old_text), _ = st.cache.popitem(last=False)
                st.stats["chars"] -= len(old_text)
            return ids

        def cached_encode_text_piece(
            self, text: str, allow_special_tokens: bool = True
        ) -> list[int]:
            key = (bool(allow_special_tokens), text)
            st = _cache(self)
            hit = lookup(self, key)
            if hit is not None:
                st.stats["hits"] += 1
                return hit.tolist()
            return insert(self, key)

        def cached_encode_chat_segments(self, segments) -> list[int]:
            st = _cache(self)
            out = array("i")
            hits = 0
            for segment in segments:
                key = (bool(segment.allow_special), segment.text)
                hit = lookup(self, key)
                if hit is None:
                    out.extend(insert(self, key))
                else:
                    hits += 1
                    out.extend(hit)
            st.stats["hits"] += hits
            return out.tolist()

        def invalidating_add_tokens(self, *args, **kwargs):
            # A vocabulary change can alter the ids of already-cached text.
            added = original_add_tokens(self, *args, **kwargs)
            st = self.__dict__.get(_CHAT_SEGMENT_CACHE_ATTR)
            if st is not None:
                st.cache.clear()
                st.stats["chars"] = 0
            return added

        setattr(tokenizer_cls, cls._ORIGINAL_TEXT_PIECE_ATTR, original)
        setattr(tokenizer_cls, cls._ORIGINAL_SEGMENTS_ATTR, original_segments)
        setattr(tokenizer_cls, cls._ORIGINAL_ADD_TOKENS_ATTR, original_add_tokens)
        tokenizer_cls._encode_text_piece = cached_encode_text_piece
        tokenizer_cls._encode_chat_segments = cached_encode_chat_segments
        tokenizer_cls.add_tokens = invalidating_add_tokens
        setattr(tokenizer_cls, cls._PATCHED_FLAG, True)
        logger.info(
            "Applying chat-segment encode cache for Kimi tokenizer "
            f"(max_chars={max_chars})"
        )
        return tokenizer

    @classmethod
    def unpatch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)
        tokenizer.__dict__.pop(_CHAT_SEGMENT_CACHE_ATTR, None)
        cls._warned_small_cap = False
        if not getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        tokenizer_cls._encode_text_piece = getattr(
            tokenizer_cls, cls._ORIGINAL_TEXT_PIECE_ATTR
        )
        tokenizer_cls._encode_chat_segments = getattr(
            tokenizer_cls, cls._ORIGINAL_SEGMENTS_ATTR
        )
        tokenizer_cls.add_tokens = getattr(tokenizer_cls, cls._ORIGINAL_ADD_TOKENS_ATTR)
        delattr(tokenizer_cls, cls._ORIGINAL_TEXT_PIECE_ATTR)
        delattr(tokenizer_cls, cls._ORIGINAL_SEGMENTS_ATTR)
        delattr(tokenizer_cls, cls._ORIGINAL_ADD_TOKENS_ATTR)
        delattr(tokenizer_cls, cls._PATCHED_FLAG)
        cls._generation += 1

        logger.info(f"Unpatched chat-segment encode cache for {tokenizer_cls.__name__}")
        return tokenizer


def _make_cached_property(cache_attr, original_fn):
    @property
    def cached_prop(self):
        if getattr(self, cache_attr, None) is None:
            setattr(self, cache_attr, original_fn(self))
        return getattr(self, cache_attr)

    return cached_prop
