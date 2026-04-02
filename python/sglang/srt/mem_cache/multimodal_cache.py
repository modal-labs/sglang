import abc
from collections import OrderedDict
from dataclasses import dataclass
import json
import logging
import time
from typing import List, Optional

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator


logger = logging.getLogger(__name__)
_MB = 1024 * 1024


class MultimodalCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
    ): ...

    @staticmethod
    def combine_hashes(mm_hashes: List[int]) -> Optional[int]:
        """
        Get a combined hash from individual mm item hashes
        """
        if not mm_hashes:
            return None
        return hash(tuple(mm_hashes))

    @abc.abstractmethod
    def get(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """
        Extract the embedding with the hash-ids of the queried items. Try combined hash first, if missed, fallback to individual hashes
        The returned tensor may not be contiguous
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set(
        self,
        mm_hash: int,
        embedding: torch.Tensor,
        mm_embedding_allocator: BaseTokenToKVPoolAllocator,
    ) -> bool:
        """
        Set the embedding to the pre-allocated locations with a hash id
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def has(self, mm_hash: int) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def available_size(self):
        raise NotImplementedError()


def _get_tensor_size(embedding: torch.Tensor):
    return embedding.element_size() * embedding.numel()


@dataclass(kw_only=True)
class EmbeddingResult:
    embedding: torch.Tensor


class MultiModalStaticCache(MultimodalCache):
    """
    A server-level cache for multimodal embedding.
    Embeddings are computed prior, and this cache does not really pre-alloc
    """

    def __init__(
        self,
        max_size: int,
    ):
        super().__init__()
        self.max_size = max_size
        self.mm_cache: OrderedDict[int, EmbeddingResult] = OrderedDict()
        self.current_size = 0
        self.get_hits_total = 0
        self.get_misses_total = 0
        self.evictions_total = 0
        self.evicted_bytes_total = 0
        self.insert_failures_total = 0
        self.insert_successes_total = 0
        self._last_log_hits = 0
        self._last_log_misses = 0
        self._last_log_evictions = 0
        self._last_log_evicted_bytes = 0
        self._last_log_insert_failures = 0
        self._last_log_insert_successes = 0
        self._last_log_ts = time.monotonic()
        self._log_interval_s = 60.0

    def _snapshot(self) -> dict:
        total_gets = self.get_hits_total + self.get_misses_total
        hit_ratio_total = (
            self.get_hits_total / total_gets if total_gets > 0 else None
        )
        fill_ratio = (
            self.current_size / self.max_size if self.max_size > 0 else None
        )
        return {
            "capacity_mb": round(self.max_size / _MB, 3),
            "used_mb": round(self.current_size / _MB, 3),
            "fill_ratio": round(fill_ratio, 4) if fill_ratio is not None else None,
            "entries": len(self.mm_cache),
            "get_hits_total": self.get_hits_total,
            "get_misses_total": self.get_misses_total,
            "hit_ratio_total": (
                round(hit_ratio_total, 4) if hit_ratio_total is not None else None
            ),
            "evictions_total": self.evictions_total,
            "evicted_mb_total": round(self.evicted_bytes_total / _MB, 3),
            "insert_failures_total": self.insert_failures_total,
            "insert_successes_total": self.insert_successes_total,
        }

    def _snapshot_deltas(self) -> dict:
        hits_delta = self.get_hits_total - self._last_log_hits
        misses_delta = self.get_misses_total - self._last_log_misses
        evictions_delta = self.evictions_total - self._last_log_evictions
        evicted_bytes_delta = self.evicted_bytes_total - self._last_log_evicted_bytes
        insert_failures_delta = (
            self.insert_failures_total - self._last_log_insert_failures
        )
        insert_successes_delta = (
            self.insert_successes_total - self._last_log_insert_successes
        )
        total_gets_delta = hits_delta + misses_delta
        hit_ratio_delta = hits_delta / total_gets_delta if total_gets_delta > 0 else None
        return {
            "hits_delta": hits_delta,
            "misses_delta": misses_delta,
            "hit_ratio_delta": (
                round(hit_ratio_delta, 4) if hit_ratio_delta is not None else None
            ),
            "evictions_delta": evictions_delta,
            "evicted_mb_delta": round(evicted_bytes_delta / _MB, 3),
            "insert_failures_delta": insert_failures_delta,
            "insert_successes_delta": insert_successes_delta,
        }

    def _mark_log_snapshot(self) -> None:
        self._last_log_hits = self.get_hits_total
        self._last_log_misses = self.get_misses_total
        self._last_log_evictions = self.evictions_total
        self._last_log_evicted_bytes = self.evicted_bytes_total
        self._last_log_insert_failures = self.insert_failures_total
        self._last_log_insert_successes = self.insert_successes_total
        self._last_log_ts = time.monotonic()

    def maybe_log_summary(self, *, reason: Optional[str] = None, force: bool = False) -> None:
        if self.max_size <= 0:
            return
        now = time.monotonic()
        if not force and (now - self._last_log_ts) < self._log_interval_s:
            return
        payload = {
            "event": "mm_embedding_cache.summary",
            "reason": reason,
            **self._snapshot(),
            **self._snapshot_deltas(),
        }
        logger.info("[mm-embedding-cache] %s", json.dumps(payload, sort_keys=True))
        self._mark_log_snapshot()

    def get(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[EmbeddingResult]:
        combined_hash = self.combine_hashes(mm_hashes)
        # MultiModalStaticCache does not fallback to individual item lookup

        embedding = self.mm_cache.get(combined_hash)
        if embedding is not None:
            self.mm_cache.move_to_end(combined_hash)
            self.get_hits_total += 1
        else:
            self.get_misses_total += 1
        self.maybe_log_summary(reason="activity")
        return embedding

    def set(
        self,
        mm_hash: int,
        embedding: EmbeddingResult,
        loc: Optional[torch.Tensor] = None,
    ) -> bool:
        assert isinstance(embedding, EmbeddingResult), embedding
        if mm_hash in self.mm_cache:
            self.mm_cache.move_to_end(mm_hash)
            self.insert_successes_total += 1
            self.maybe_log_summary(reason="activity")
            return True
        data_size = _get_tensor_size(embedding.embedding)
        while self.current_size + data_size > self.max_size:
            if not self.mm_cache:
                self.insert_failures_total += 1
                self.maybe_log_summary(reason="insert_failure", force=True)
                return False
            lru_hash, lru_embedding = self.mm_cache.popitem(last=False)
            evicted_size = _get_tensor_size(lru_embedding.embedding)
            self.current_size -= evicted_size
            self.evictions_total += 1
            self.evicted_bytes_total += evicted_size

        self.mm_cache[mm_hash] = embedding
        self.current_size += data_size
        self.insert_successes_total += 1
        self.maybe_log_summary(reason="activity")
        return True

    def get_single(self, mm_hash: int) -> Optional[EmbeddingResult]:
        """Get a single cached embedding by its hash (no combine_hashes)."""
        embedding = self.mm_cache.get(mm_hash)
        if embedding is not None:
            self.mm_cache.move_to_end(mm_hash)
        return embedding

    def has(self, mm_hash: int) -> bool:
        return mm_hash in self.mm_cache

    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        if mm_hash not in self.mm_cache:
            return False
        old_embedding = self.mm_cache.pop(mm_hash)
        self.current_size -= _get_tensor_size(old_embedding.embedding)
        return True

    def clear(self):
        self.mm_cache.clear()
        self.current_size = 0
        self.maybe_log_summary(reason="clear", force=True)

    def __len__(self):
        return len(self.mm_cache)

    def available_size(self):
        return self.__len__()
