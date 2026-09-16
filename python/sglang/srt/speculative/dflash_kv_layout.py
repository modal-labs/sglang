"""Layout accounting for a TP-sharded draft with a DCP-sharded target."""


def draft_kv_bytes_per_target_token(
    *,
    num_layers: int,
    total_kv_heads: int,
    head_dim: int,
    tp_size: int,
    dcp_size: int,
    dtype: str,
) -> int:
    """Draft bytes per physical target row, covering DCP logical token slots.

    The draft partitions heads by TP and replicates the target's token domain.
    Each physical target row therefore backs dcp_size draft rows on each rank.
    """
    if min(num_layers, total_kv_heads, head_dim, tp_size, dcp_size) <= 0:
        raise ValueError("Draft KV dimensions and parallel sizes must be positive")
    if total_kv_heads >= tp_size and total_kv_heads % tp_size:
        raise ValueError("Draft KV heads must divide evenly across TP ranks")
    if total_kv_heads < tp_size and tp_size % total_kv_heads:
        raise ValueError("TP ranks must divide evenly across replicated KV heads")
    dtype = (dtype or "").lower()
    if dtype in ("bf16", "bfloat16", "fp16", "float16"):
        element_bytes = 2
    elif dtype == "float32":
        element_bytes = 4
    elif dtype in ("fp8_e4m3", "fp8_e5m2", "fp8", "mxfp8"):
        element_bytes = 1
    else:
        raise ValueError(f"Unresolved draft KV dtype: {dtype}")
    return (
        num_layers
        * 2
        * max(1, total_kv_heads // tp_size)
        * head_dim
        * element_bytes
        * dcp_size
    )


def draft_kv_transfer_start(
    seq_len: int, window_size: int | None, page_size: int
) -> int:
    """First logical token position needed by the draft, rounded to a page."""
    if seq_len < 0 or page_size <= 0 or (window_size is not None and window_size <= 0):
        raise ValueError("Invalid draft transfer extent")
    start = max(0, seq_len - window_size) if window_size is not None else 0
    return start // page_size * page_size
