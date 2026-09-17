import torch

FP8_SATFINITE_KINDS = ("gt448", "gt464", "nonfinite")


class Fp8SatfiniteTelemetry:
    """Per-layer on-device counters of rows that to_fp8_satfinite will clamp."""

    def __init__(self, num_layers: int, device, every: int, thresholds=(448.0, 464.0)):
        self.every = every
        self.thresholds = thresholds
        self.counts = torch.zeros(
            (num_layers, len(FP8_SATFINITE_KINDS)), dtype=torch.int64, device=device
        )
        self.amax = torch.zeros(num_layers, dtype=torch.float32, device=device)
        self._chunk_idx = 0
        self._sampling = False
        self._dirty = False
        self._first_layer_id = None

    def begin_chunk(self, layer_id: int) -> None:
        """Advance the chunk counter once per forward (on the first layer seen)."""
        if self._first_layer_id is None:
            self._first_layer_id = layer_id
        if layer_id == self._first_layer_id:
            self._chunk_idx += 1
            self._sampling = self.every > 0 and self._chunk_idx % self.every == 0

    def should_record(self, layer_id: int) -> bool:
        self.begin_chunk(layer_id)
        return self._sampling

    def record(self, layer_id: int, x: torch.Tensor) -> None:
        row_amax = x.detach().reshape(x.shape[0], -1).abs().amax(dim=1).float()
        finite = torch.isfinite(row_amax)
        finite_amax = torch.where(finite, row_amax, torch.zeros_like(row_amax))
        c = self.counts[layer_id]
        c[0] += (finite & (row_amax > self.thresholds[0])).sum()
        c[1] += (finite & (row_amax > self.thresholds[1])).sum()
        c[2] += (~finite).sum()
        self.amax[layer_id] = torch.maximum(self.amax[layer_id], finite_amax.max())
        self._dirty = True

    def drain(self) -> list[tuple[int, str, int, float]]:
        """One host sync; returns (layer_id, kind, count, layer_amax) for
        nonzero counts and zeroes the buffers."""
        if not self._dirty:
            return []
        self._dirty = False
        counts = self.counts.tolist()
        amax = self.amax.tolist()
        self.counts.zero_()
        self.amax.zero_()
        out = []
        for layer_id, row in enumerate(counts):
            for kind, n in zip(FP8_SATFINITE_KINDS, row):
                if n:
                    out.append((layer_id, kind, n, amax[layer_id]))
        return out
