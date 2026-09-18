"""Ghost list of evicted KV pages: the capacity-miss signal for the radix caches.

The KV age histograms say how old a node was when it was matched or evicted.
They cannot say whether an eviction *mattered*: whether the traffic came back
for that prefix and had to recompute it. This module answers that directly,
the way ARC's ghost lists do, but purely as a measurement:

* when a radix node's last copy is destroyed (a ``dropped`` eviction on any
  tier), the per-page hashes of its prefix are remembered with the eviction
  time in a bounded, time-limited dict;
* when a node is created by an insert, its page hashes are looked up. A hit
  means the cache used to hold exactly this prefix and a request needed it
  again: the tokens were recomputed because of eviction, and ``now - evicted``
  is how much longer the tier would have needed to keep them.

Hashes are chained per page from a namespace seed (``extra_key`` and
``cache_salt``), so a page's digest is a function of the whole prefix up to
that page, not of the node that happens to hold it. Splitting a node just
partitions its digest list, and a dropped-and-reinserted prefix reproduces the
same digests from a fresh tree. The chain is computed with stdlib
``hashlib.blake2b`` (8-byte digests) rather than the HiCache native hash, so it
carries no build dependency and about 1.5 us per 64-token page.

Everything here runs on the scheduler thread with the cache; there is no
locking. The list is created only when ``--enable-metrics`` is on (see
``build_kv_ghost_tracker``), bounded by ``SGLANG_KV_GHOST_LIST_PAGES`` entries
and ``SGLANG_KV_GHOST_LIST_TTL_S`` seconds.
"""

from __future__ import annotations

import hashlib
import logging
import time
from array import array
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional, Sequence, Tuple

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.observability.metrics_collector import RadixCacheMetricsCollector

logger = logging.getLogger(__name__)

GHOST_DIGEST_BYTES = 8
_ROOT_SEED = bytes(GHOST_DIGEST_BYTES)
# Auto-sizing cap: the list tracks at most this many pages regardless of the
# KV pool (1M entries is ~100 MB of dict).
GHOST_LIST_MAX_AUTO_PAGES = 1 << 20


def ghost_namespace_seed(extra_key: Optional[str], cache_salt: Optional[str]) -> bytes:
    """Seed of a prefix chain: the root's digest for a (extra_key, cache_salt)
    namespace. Presence and byte length distinguish absent, empty and joined
    parts, like the storage namespace seed."""
    if extra_key is None and cache_salt is None:
        return _ROOT_SEED
    digest = hashlib.blake2b(
        b"sglang-kv-ghost-namespace-v1", digest_size=GHOST_DIGEST_BYTES
    )
    for part in (extra_key, cache_salt):
        if part is None:
            digest.update(b"\x00")
            continue
        encoded = part.encode("utf-8")
        digest.update(b"\x01" + len(encoded).to_bytes(8, "little") + encoded)
    return digest.digest()


def _raw_bytes(key: Any) -> bytes:
    token_ids = key.token_ids
    if not isinstance(token_ids, array):
        token_ids = array("q", token_ids)
    return token_ids.tobytes()


def chain_page_digests(key: Any, prior: bytes, page_size: int) -> List[int]:
    """Per-page digests of ``key`` chained from ``prior``.

    Page ``i``'s digest covers ``prior`` and the page's raw tokens, so through
    the chain it identifies the entire prefix ending at that page. Returned as
    ints (little-endian) because they are dict keys; the node stores them and
    hands the last one to its children as their ``prior``.
    """
    n = len(key)
    if page_size > 1:
        n = n // page_size * page_size
    digests: List[int] = []
    for start in range(0, n, page_size):
        prior = hashlib.blake2b(
            prior + _raw_bytes(key[start : start + page_size]),
            digest_size=GHOST_DIGEST_BYTES,
        ).digest()
        digests.append(int.from_bytes(prior, "little"))
    return digests


def ensure_ghost_digests(node: Any, page_size: int) -> List[int]:
    """``node.ghost_hash``, computing it (and any missing ancestors') from the
    root chain. Nodes created on paths that do not hook the tracker (host-side
    inserts, load-backs) are filled in lazily here."""
    digests = node.ghost_hash
    if digests is not None:
        return digests
    missing = []
    cur = node
    while cur is not None and cur.ghost_hash is None:
        missing.append(cur)
        cur = cur.parent
    # ``cur`` is the deepest ancestor that already has digests (the root always
    # does; a detached node with no root above it gets the namespace seed).
    prior_digests = cur.ghost_hash if cur is not None else []
    for cur in reversed(missing):
        if cur.parent is None or cur.key is None:
            cur.ghost_hash = []
            prior_digests = cur.ghost_hash
            continue
        if prior_digests:
            prior = prior_digests[-1].to_bytes(GHOST_DIGEST_BYTES, "little")
        else:
            prior = ghost_namespace_seed(cur.key.extra_key, cur.key.cache_salt)
        cur.ghost_hash = chain_page_digests(cur.key, prior, page_size)
        prior_digests = cur.ghost_hash
    return node.ghost_hash


class GhostHit(NamedTuple):
    """A run of consecutive reinserted pages that left the cache together."""

    delay_seconds: float
    num_pages: int


class KVGhostList:
    """Bounded LRU of page digests evicted from the cache, keyed digest ->
    eviction time (``time.monotonic``). Oldest first."""

    def __init__(self, capacity: int, ttl_seconds: float):
        assert capacity > 0 and ttl_seconds > 0
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[int, float] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def record(self, digests: Sequence[int], now: float) -> Tuple[int, int]:
        """Remember ``digests`` as evicted at ``now``.

        Returns ``(expired_ttl, expired_capacity)``: entries discarded because
        they were older than the TTL, and because the list was full.
        """
        entries = self._entries
        cutoff = now - self.ttl_seconds
        expired_ttl = 0
        while entries:
            _, oldest = next(iter(entries.items()))
            if oldest >= cutoff:
                break
            entries.popitem(last=False)
            expired_ttl += 1
        for digest in digests:
            if digest in entries:
                # Evicted again without an intervening hooked insert (e.g. it
                # came back through a host-side path): keep the latest time.
                entries.move_to_end(digest)
            entries[digest] = now
        expired_capacity = 0
        while len(entries) > self.capacity:
            entries.popitem(last=False)
            expired_capacity += 1
        return expired_ttl, expired_capacity

    def match(self, digests: Sequence[int], now: float) -> List[GhostHit]:
        """Pop every digest present and group consecutive pages that share an
        eviction time into one hit (one dropped node fragment came back).
        Entries past the TTL that ``record`` has not purged yet do not count."""
        entries = self._entries
        cutoff = now - self.ttl_seconds
        hits: List[GhostHit] = []
        run_time = 0.0
        run_pages = 0
        for digest in digests:
            evicted_at = entries.pop(digest, None)
            if evicted_at is None or evicted_at < cutoff:
                if run_pages:
                    hits.append(GhostHit(now - run_time, run_pages))
                    run_pages = 0
                continue
            if run_pages and evicted_at == run_time:
                run_pages += 1
                continue
            if run_pages:
                hits.append(GhostHit(now - run_time, run_pages))
            run_time = evicted_at
            run_pages = 1
        if run_pages:
            hits.append(GhostHit(now - run_time, run_pages))
        return hits


class KVGhostTracker:
    """Binds a ``KVGhostList`` to a cache's page size and metrics collector.

    ``on_dropped`` is called once per node whose last copy is destroyed;
    ``on_inserted`` once per node an insert creates. Both take the node (any
    object with ``key``, ``parent`` and ``ghost_hash``) and an optional
    pre-read ``time.monotonic()``.
    """

    def __init__(
        self,
        ghost_list: KVGhostList,
        page_size: int,
        metrics: RadixCacheMetricsCollector,
    ):
        self.ghost_list = ghost_list
        self.page_size = page_size
        self.metrics = metrics

    def reset(self) -> None:
        """The tree was flushed: its ghosts would otherwise read as capacity
        misses when the same prompts come back."""
        n = len(self.ghost_list)
        self.ghost_list.clear()
        if n:
            self.metrics.increment_kv_ghost_list_discarded("flush", n)

    def on_dropped(self, node: Any, now: Optional[float] = None) -> None:
        if node.parent is None or node.key is None:
            return
        digests = ensure_ghost_digests(node, self.page_size)
        if not digests:
            return
        if now is None:
            now = time.monotonic()
        expired_ttl, expired_capacity = self.ghost_list.record(digests, now)
        metrics = self.metrics
        metrics.increment_kv_ghost_list_recorded(len(digests))
        if expired_ttl:
            metrics.increment_kv_ghost_list_discarded("ttl", expired_ttl)
        if expired_capacity:
            metrics.increment_kv_ghost_list_discarded("capacity", expired_capacity)

    def on_inserted(self, node: Any, now: Optional[float] = None) -> None:
        if node.parent is None or node.key is None:
            return
        digests = ensure_ghost_digests(node, self.page_size)
        if not digests:
            return
        if now is None:
            now = time.monotonic()
        metrics = self.metrics
        metrics.increment_kv_inserted_tokens(len(digests) * self.page_size)
        hits = self.ghost_list.match(digests, now)
        if not hits:
            return
        popped = 0
        for hit in hits:
            metrics.observe_kv_reinsert_after_evict(
                hit.delay_seconds, hit.num_pages * self.page_size
            )
            popped += hit.num_pages
        metrics.increment_kv_ghost_list_discarded("reinsert", popped)


def resolve_kv_ghost_list_pages(
    allocator: Any, page_size: int, requested: Optional[int] = None
) -> int:
    """Entry cap for the ghost list: ``SGLANG_KV_GHOST_LIST_PAGES`` when set
    (0 disables), else the KV pool's page count (ARC sizing: remember as much
    as the cache holds), capped at ``GHOST_LIST_MAX_AUTO_PAGES``."""
    pages = int(
        envs.SGLANG_KV_GHOST_LIST_PAGES.get() if requested is None else requested
    )
    if pages >= 0:
        return pages
    pool_tokens = getattr(allocator, "size", None)
    if not isinstance(pool_tokens, int) or pool_tokens <= 0:
        return GHOST_LIST_MAX_AUTO_PAGES
    return max(1, min(pool_tokens // max(page_size, 1), GHOST_LIST_MAX_AUTO_PAGES))


def build_kv_ghost_tracker(
    allocator: Any,
    page_size: int,
    metrics: Optional[RadixCacheMetricsCollector],
) -> Optional[KVGhostTracker]:
    """The tracker a cache installs when metrics are on, or None when they are
    off or ``SGLANG_KV_GHOST_LIST_PAGES=0``."""
    if metrics is None:
        return None
    pages = resolve_kv_ghost_list_pages(allocator, page_size)
    ttl = envs.SGLANG_KV_GHOST_LIST_TTL_S.get()
    if pages <= 0 or ttl <= 0:
        return None
    logger.info(
        "KV ghost list enabled: %d pages of %d tokens, TTL %.0f s",
        pages,
        page_size,
        ttl,
    )
    return KVGhostTracker(KVGhostList(pages, ttl), page_size, metrics)
