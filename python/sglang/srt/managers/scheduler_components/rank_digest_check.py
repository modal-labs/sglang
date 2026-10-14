"""Per-rank batch digest check for TP extend forwards.

Gate: SGLANG_K3_RANK_DIGEST_CHECK=1 (default off).

K3 prod prefill gangs (TP8, DCP8, HiCache V3 with rank-0 authority) wedge with
every rank spinning in a collective mid-forward. The number and sizes of the
DCP KV all-gathers inside the chunked-prefix MLA attention loop derive from
each rank's own ``extend_prefix_lens_cpu`` / ``num_prefix_chunks``
(``prepare_chunked_prefix_cache_info``), and HiCache V3 assumes every rank
computes identical values ("no load-initiation consensus"). NCCL hangs rather
than faults when they differ, so a divergence is indistinguishable from any
other hang.

Once per extend forward, before the model forward launches, every TP rank
builds a small CPU digest of its ScheduleBatch (request ids in order,
extend_lens, extend_prefix_lens, seq_lens, the eager attention forward-method
decision, the chunked-prefix plan, and a running forward counter), encodes it
as a fixed-size int64 tensor, all-gathers it over the TP gloo group with a
bounded wait, and compares. The collective itself can never mismatch: every
rank contributes the same shape regardless of its batch. On any mismatch each
rank logs one ERROR line with the differing fields and every rank's values,
then raises RuntimeError, the same fail-stop the HiCache V3 CRC divergence path
uses, so the gang dies with a diagnosis instead of a silent wedge.

Cost: one gloo all-gather of ``8 + 4 * max_batch`` int64 per extend batch and
no device work.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import timedelta
from enum import IntEnum
from typing import Callable, List, Optional, Sequence, Tuple

import msgspec
import torch

logger = logging.getLogger(__name__)

# Encoded layout: a fixed header followed by four vectors of ``max_batch``
# int64 each, padded with _PAD. The hash in slot 0 covers everything in the
# digest (including the chunk matrix, which is gathered only as its own hash).
_HEADER = (
    "digest_hash",
    "forward_ct",
    "batch_size",
    "forward_mode",
    "attn_forward_method",
    "prefix_chunk_len",
    "num_prefix_chunks",
    "prefix_chunk_seq_lens_hash",
)
_VECTORS = ("rid_hashes", "extend_lens", "extend_prefix_lens", "seq_lens")
_PAD = -1


class DigestAttnMethod(IntEnum):
    """Eager attention forward-method decision mirrored for the digest.

    Values are local to the digest (not AttnForwardMethod) so this module
    stays importable without the model stack.
    """

    UNMIRRORED = 0
    MLA = 1
    MHA_CHUNKED_KV = 2
    MHA_ONE_SHOT = 3


class AttnDecisionConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Init-static inputs of the eager attention dispatch (deepseek_common/
    attention_backend_handler.py), captured once at scheduler init."""

    prefill_backend: str
    disable_chunked_prefix_cache: bool
    chunked_prefix_cache_threshold: int
    flashinfer_mla_disable_ragged: bool
    enable_deterministic_inference: bool
    mla_prefill_cp: bool
    chunk_capacity: int


class RankBatchDigest(msgspec.Struct, frozen=True, kw_only=True):
    forward_ct: int
    batch_size: int
    forward_mode: int
    rids: Tuple[str, ...]
    rid_hashes: Tuple[int, ...]
    extend_lens: Tuple[int, ...]
    extend_prefix_lens: Tuple[int, ...]
    seq_lens: Tuple[int, ...]
    attn_forward_method: int
    prefix_chunk_len: int
    num_prefix_chunks: int
    prefix_chunk_seq_lens: Tuple[Tuple[int, ...], ...]


_ONE_SHOT_BACKENDS = ("fa3", "flashinfer", "flashmla")
_TRTLLM_STYLE_BACKENDS = ("trtllm_mla", "tokenspeed_mla")
_GENERIC_BACKENDS = ("fa3", "fa4", "flashinfer", "flashmla", "cutlass_mla")


def _hash64(payload: bytes) -> int:
    """Process-independent 64-bit hash (str hash() is salted per process)."""
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "little", signed=True
    )


def hash_rid(rid: str) -> int:
    return _hash64(rid.encode("utf-8", errors="surrogatepass"))


def decide_extend_attn_forward_method(
    *,
    cfg: AttnDecisionConfig,
    is_extend_without_speculative: bool,
    sum_extend_prefix_lens: int,
    sum_seq_lens: int,
) -> DigestAttnMethod:
    """Mirror of the eager (no captured-graph context) dispatch in
    attention_backend_handler.py for the DeepSeek MLA backends. The value is
    compared across ranks, not fed back into the forward, so an unmirrored
    backend simply contributes UNMIRRORED on every rank."""
    backend = cfg.prefill_backend
    if cfg.mla_prefill_cp:
        return DigestAttnMethod.MLA
    if backend in _TRTLLM_STYLE_BACKENDS:
        if is_extend_without_speculative and (
            not cfg.disable_chunked_prefix_cache or sum_extend_prefix_lens == 0
        ):
            return DigestAttnMethod.MHA_CHUNKED_KV
        return DigestAttnMethod.MLA
    if backend not in _GENERIC_BACKENDS:
        return DigestAttnMethod.UNMIRRORED
    if backend in ("fa3", "fa4") and cfg.enable_deterministic_inference:
        return DigestAttnMethod.MLA
    disable_ragged = (
        backend in ("flashinfer", "flashmla") and cfg.flashinfer_mla_disable_ragged
    )
    if (
        not disable_ragged
        and is_extend_without_speculative
        and (
            (
                sum_extend_prefix_lens >= cfg.chunked_prefix_cache_threshold
                and not cfg.disable_chunked_prefix_cache
            )
            or sum_extend_prefix_lens == 0
        )
    ):
        if backend in _ONE_SHOT_BACKENDS and sum_seq_lens <= cfg.chunk_capacity:
            return DigestAttnMethod.MHA_ONE_SHOT
        return DigestAttnMethod.MHA_CHUNKED_KV
    return DigestAttnMethod.MLA


def chunked_prefix_plan(
    *, extend_prefix_lens: Sequence[int], chunk_capacity: int
) -> Tuple[int, int, Tuple[Tuple[int, ...], ...]]:
    """Pure-Python mirror of ForwardBatchDeepSeekMHAMixin.
    prepare_chunked_prefix_cache_info + get_prefix_chunk_seq_lens, the values
    that size the per-chunk DCP KV all-gathers.

    Returns (prefix_chunk_len, num_prefix_chunks, prefix_chunk_seq_lens) with
    prefix_chunk_seq_lens shaped (num_prefix_chunks, batch_size).
    """
    batch_size = len(extend_prefix_lens)
    if batch_size == 0 or not any(extend_prefix_lens):
        return 0, 0, ()
    prefix_chunk_len = chunk_capacity // batch_size
    num_prefix_chunks = (
        max(extend_prefix_lens) + prefix_chunk_len - 1
    ) // prefix_chunk_len
    rows = []
    for idx in range(num_prefix_chunks):
        start = idx * prefix_chunk_len
        rows.append(
            tuple(
                max(min(prefix_len, start + prefix_chunk_len) - start, 0)
                for prefix_len in extend_prefix_lens
            )
        )
    return prefix_chunk_len, num_prefix_chunks, tuple(rows)


def build_digest(
    *,
    forward_ct: int,
    forward_mode: int,
    is_extend_without_speculative: bool,
    rids: Sequence[str],
    extend_lens: Sequence[int],
    extend_prefix_lens: Sequence[int],
    seq_lens: Sequence[int],
    cfg: AttnDecisionConfig,
) -> RankBatchDigest:
    extend_prefix_lens = tuple(int(x) for x in extend_prefix_lens)
    seq_lens = tuple(int(x) for x in seq_lens)
    method = decide_extend_attn_forward_method(
        cfg=cfg,
        is_extend_without_speculative=is_extend_without_speculative,
        sum_extend_prefix_lens=sum(extend_prefix_lens),
        sum_seq_lens=sum(seq_lens),
    )
    if method == DigestAttnMethod.MHA_CHUNKED_KV:
        prefix_chunk_len, num_prefix_chunks, chunk_rows = chunked_prefix_plan(
            extend_prefix_lens=extend_prefix_lens, chunk_capacity=cfg.chunk_capacity
        )
    else:
        # MHA_ONE_SHOT pins num_prefix_chunks = 0; MLA never prepares chunks.
        prefix_chunk_len, num_prefix_chunks, chunk_rows = 0, 0, ()
    return RankBatchDigest(
        forward_ct=forward_ct,
        batch_size=len(rids),
        forward_mode=forward_mode,
        rids=tuple(rids),
        rid_hashes=tuple(hash_rid(rid) for rid in rids),
        extend_lens=tuple(int(x) for x in extend_lens),
        extend_prefix_lens=extend_prefix_lens,
        seq_lens=seq_lens,
        attn_forward_method=int(method),
        prefix_chunk_len=prefix_chunk_len,
        num_prefix_chunks=num_prefix_chunks,
        prefix_chunk_seq_lens=chunk_rows,
    )


def encoded_length(max_batch: int) -> int:
    return len(_HEADER) + len(_VECTORS) * max_batch


def encode_digest(digest: RankBatchDigest, *, max_batch: int) -> torch.Tensor:
    """Fixed-size int64 encoding; vectors longer than max_batch are truncated
    (the digest hash still covers them)."""
    header = [
        _hash64(msgspec.msgpack.encode(digest)),
        digest.forward_ct,
        digest.batch_size,
        digest.forward_mode,
        digest.attn_forward_method,
        digest.prefix_chunk_len,
        digest.num_prefix_chunks,
        _hash64(msgspec.msgpack.encode(digest.prefix_chunk_seq_lens)),
    ]
    out = torch.full((encoded_length(max_batch),), _PAD, dtype=torch.int64)
    out[: len(header)] = torch.tensor(header, dtype=torch.int64)
    for i, name in enumerate(_VECTORS):
        values = list(getattr(digest, name))[:max_batch]
        if values:
            offset = len(_HEADER) + i * max_batch
            out[offset : offset + len(values)] = torch.tensor(values, dtype=torch.int64)
    return out


def decode_fields(encoded: torch.Tensor, *, max_batch: int) -> dict:
    values = encoded.tolist()
    fields = {name: values[i] for i, name in enumerate(_HEADER)}
    for i, name in enumerate(_VECTORS):
        offset = len(_HEADER) + i * max_batch
        vec = values[offset : offset + max_batch]
        fields[name] = [v for v in vec if v != _PAD]
    return fields


def compare_digests(
    gathered: Sequence[torch.Tensor], *, max_batch: int
) -> List[Tuple[str, List]]:
    """Fields whose value is not identical on every rank, each with the
    per-rank values in rank order. Empty when all ranks agree."""
    per_rank = [decode_fields(t, max_batch=max_batch) for t in gathered]
    if not per_rank or all(
        r["digest_hash"] == per_rank[0]["digest_hash"] for r in per_rank
    ):
        return []
    mismatches = []
    for name in _HEADER + _VECTORS:
        if name == "digest_hash":
            continue
        values = [r[name] for r in per_rank]
        if any(v != values[0] for v in values):
            mismatches.append((name, values))
    if not mismatches:
        # Hash differs but no gathered field does: the difference is in a
        # field only the hash covers (truncated vectors or rid strings whose
        # hashes collide); surface the hashes so the log is never empty.
        mismatches.append(("digest_hash", [r["digest_hash"] for r in per_rank]))
    return mismatches


def gloo_all_gather(
    group, *, world_size: int, timeout_s: float
) -> Callable[[torch.Tensor], List[torch.Tensor]]:
    """All-gather over a gloo process group with a bounded wait. A rank that
    never reaches the gather (it is running a different batch, or is already
    wedged) surfaces as a RuntimeError here instead of an unbounded hang."""

    def _gather(tensor: torch.Tensor) -> List[torch.Tensor]:
        out = [torch.empty_like(tensor) for _ in range(world_size)]
        work = torch.distributed.all_gather(out, tensor, group=group, async_op=True)
        work.wait(timeout=timedelta(seconds=timeout_s))
        return out

    return _gather


class RankDigestChecker:
    """Runs the digest exchange once per extend forward on every TP rank."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        max_batch: int,
        cfg: AttnDecisionConfig,
        all_gather: Callable[[torch.Tensor], List[torch.Tensor]],
    ):
        self.rank = rank
        self.world_size = world_size
        self.max_batch = max_batch
        self.cfg = cfg
        self._all_gather = all_gather
        self.forward_ct = 0
        self.last_digest: Optional[RankBatchDigest] = None

    def check(
        self,
        *,
        forward_mode: int,
        is_extend_without_speculative: bool,
        rids: Sequence[str],
        extend_lens: Sequence[int],
        extend_prefix_lens: Sequence[int],
        seq_lens_cpu: Optional[torch.Tensor],
    ) -> RankBatchDigest:
        self.forward_ct += 1
        # merge_batch drops seq_lens_cpu when either side lacks it; an absent
        # vector is itself part of the digest (empty on every rank, or not).
        seq_lens = seq_lens_cpu.tolist() if seq_lens_cpu is not None else []
        digest = build_digest(
            forward_ct=self.forward_ct,
            forward_mode=forward_mode,
            is_extend_without_speculative=is_extend_without_speculative,
            rids=rids,
            extend_lens=extend_lens,
            extend_prefix_lens=extend_prefix_lens,
            seq_lens=seq_lens,
            cfg=self.cfg,
        )
        self.last_digest = digest
        encoded = encode_digest(digest, max_batch=self.max_batch)
        try:
            gathered = self._all_gather(encoded)
        except RuntimeError as e:
            raise RuntimeError(
                f"K3 rank digest divergence: rank {self.rank} extend forward "
                f"#{self.forward_ct} digest all-gather did not complete "
                f"({e}); peers are not running this batch. "
                f"local={self._describe(digest)}"
            ) from e
        mismatches = compare_digests(gathered, max_batch=self.max_batch)
        if not mismatches:
            return digest
        detail = "; ".join(
            f"{name}: "
            + " ".join(f"r{r}={_fmt(name, v)}" for r, v in enumerate(values))
            for name, values in mismatches
        )
        logger.error(
            "K3 rank digest divergence at extend forward #%d on rank %d: %s; "
            "local=%s",
            self.forward_ct,
            self.rank,
            detail,
            self._describe(digest),
        )
        raise RuntimeError(
            f"K3 rank digest divergence at extend forward #{self.forward_ct} "
            f"on rank {self.rank}: {[name for name, _ in mismatches]} differ "
            "across TP ranks (fail-stop; see the ERROR line above)."
        )

    @staticmethod
    def _describe(digest: RankBatchDigest) -> str:
        return (
            f"rids={list(digest.rids)} "
            f"attn_forward_method={DigestAttnMethod(digest.attn_forward_method).name} "
            f"prefix_chunk_seq_lens={[list(r) for r in digest.prefix_chunk_seq_lens]}"
        )


def _fmt(name: str, value) -> str:
    if isinstance(value, list):
        return "[" + ",".join(str(v) for v in value) + "]"
    if name == "attn_forward_method":
        return DigestAttnMethod(value).name
    return str(value)
