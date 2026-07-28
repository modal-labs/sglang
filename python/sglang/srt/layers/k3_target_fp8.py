"""Online static-FP8 replacement for selected Kimi-K3 target linears.

K3 checkpoints mix BF16 dense weights with MXFP4 routed experts, so applying a
global FP8 config would interfere with model-specific post-load fusions. This
module instead converts only explicitly selected, already-merged target
linears.

BF16 source tensors live in one private CUDA MemPool per conversion unit.
Persistent FP8 weights and scales are allocated after leaving that pool, in
PyTorch's normal allocator. Once every BF16 alias has been replaced,
``release`` destroys the private pool and returns its segments to CUDA without
requiring expandable segments.

Two static E4M3 representations are selectable:

* ``tensor_static``: one weight scale and unit activation scale, dispatched
  through ``torch._scaled_mm``. This is the latency-oriented default.
* ``channel_static``: one scale per output channel and unit activation scale,
  dispatched through SGLang's CUTLASS FP8 scaled GEMM. This is the
  range-robust correctness fallback.

The wrapper deliberately does not expose a ``quant_method`` attribute. Model
loaders traverse such attributes after ``post_load_weights``; exposing one
would process an already-converted weight a second time.
"""

from __future__ import annotations

import gc
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
from torch import nn

from sglang.kernels.ops.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
)
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    input_to_float8,
)
from sglang.srt.utils import rank0_log

logger = logging.getLogger(__name__)

_SUPPORTED_SCOPES = frozenset({"off", "front", "wide"})
_SUPPORTED_REPRESENTATIONS = frozenset({"tensor_static", "channel_static"})
_FRONT_ROLE = "moe_front"
_KDA_QKVG_ROLE = "kda_qkvg"
_EMPTY_DEFAULT_CACHE_EVERY_GROUPS = 8
_CHECKPOINT_COMPONENTS_BY_ROLE = {
    _FRONT_ROLE: frozenset(
        {
            "shared_gate",
            "shared_up",
            "router",
            "latent_down",
        }
    ),
    _KDA_QKVG_ROLE: frozenset({"q", "k", "v", "g"}),
}


def _cuda_memory_snapshot() -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    stats = torch.cuda.memory_stats()
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(stats.get("allocated_bytes.all.current", 0)),
        "reserved_bytes": int(stats.get("reserved_bytes.all.current", 0)),
        "inactive_split_bytes": int(stats.get("inactive_split_bytes.all.current", 0)),
    }


def rebind_weight_aliases(
    modules: Sequence[nn.Module],
    row_sizes: Sequence[int],
    logical_replacement: torch.Tensor,
) -> None:
    """Rebind component Parameters to row views of one logical [N, K] FP8 weight."""
    if len(modules) != len(row_sizes):
        raise ValueError("K3 target FP8 alias module/row counts do not match.")
    if sum(row_sizes) != logical_replacement.shape[0]:
        raise ValueError(
            "K3 target FP8 alias rows do not cover the replacement: "
            f"rows={sum(row_sizes)}, replacement={logical_replacement.shape[0]}."
        )
    off = 0
    for module, rows in zip(modules, row_sizes):
        weight = getattr(module, "weight", None)
        if not isinstance(weight, nn.Parameter):
            raise TypeError(f"{type(module).__name__} has no weight Parameter.")
        weight.requires_grad_(False)
        weight.data = logical_replacement.data[off : off + rows]
        off += rows


@dataclass
class K3TargetFP8MemoryStats:
    """Exact logical resident bytes for one TP rank."""

    scope: str
    representation: str
    staged_source_bytes: int = 0
    replaced_source_bytes: int = 0
    final_weight_bytes: int = 0
    final_scale_bytes: int = 0
    replaced_linears: int = 0

    @property
    def final_bytes(self) -> int:
        return self.final_weight_bytes + self.final_scale_bytes

    @property
    def saved_bytes(self) -> int:
        return self.replaced_source_bytes - self.final_bytes

    def as_dict(self) -> dict[str, int | str]:
        result = asdict(self)
        result["final_bytes"] = self.final_bytes
        result["saved_bytes"] = self.saved_bytes
        return result


class K3TargetFP8Linear(nn.Module):
    """Static E4M3 linear with a unit activation scale."""

    def __init__(
        self,
        runtime_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        representation: str,
    ) -> None:
        super().__init__()
        if representation not in _SUPPORTED_REPRESENTATIONS:
            raise ValueError(
                f"Unsupported K3 target FP8 representation {representation!r}."
            )
        self.representation = representation
        self.register_parameter(
            "weight",
            nn.Parameter(runtime_weight.detach(), requires_grad=False),
        )
        self.register_parameter(
            "weight_scale",
            nn.Parameter(weight_scale.detach(), requires_grad=False),
        )
        self.register_parameter(
            "input_scale",
            nn.Parameter(
                torch.ones(
                    1,
                    dtype=torch.float32,
                    device=runtime_weight.device,
                ),
                requires_grad=False,
            ),
        )

    @classmethod
    def from_bf16(
        cls,
        source: torch.Tensor,
        *,
        representation: str,
    ) -> K3TargetFP8Linear:
        if not source.is_cuda:
            raise RuntimeError(
                "K3 target FP8 conversion requires a CUDA-resident source; "
                "CPU/offloaded weights cannot use the private-pool HBM path."
            )
        if source.ndim != 2 or not source.is_contiguous():
            raise ValueError("K3 target FP8 requires a contiguous 2D source weight.")
        if source.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(
                f"K3 target FP8 source must be BF16/FP16, got {source.dtype}."
            )
        n, k = source.shape
        if n % 16 or k % 16:
            raise ValueError(
                "K3 target FP8 requires N and K divisible by 16, "
                f"got shape={tuple(source.shape)}."
            )

        if representation == "tensor_static":
            logical_weight, weight_scale = input_to_float8(source)
        elif representation == "channel_static":
            logical_weight, weight_scale = per_token_group_quant_fp8(
                source,
                group_size=k,
            )
            weight_scale = weight_scale.t().contiguous()
        else:
            raise ValueError(
                f"Unsupported K3 target FP8 representation {representation!r}."
            )

        # apply_fp8_linear consumes the runtime weight as [K, N]. Keep the
        # transpose as a view so component modules can alias weight.t() and
        # retain their checkpoint-facing [N, K] shapes without another copy.
        return cls(
            logical_weight.t(),
            weight_scale,
            representation=representation,
        )

    @property
    def logical_weight(self) -> torch.Tensor:
        """The shared [N, K] view used only for structural component aliases."""
        return self.weight.t()

    @property
    def resident_scale_bytes(self) -> int:
        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in (self.weight_scale, self.input_scale)
        )

    @property
    def supports_prequantized_static_input(self) -> bool:
        """Whether the linear can consume the AttnRes unit-scale FP8 output."""
        return self.representation == "tensor_static"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_empty((*x.shape[:-1], self.weight.shape[1]))
        use_channel_cutlass = self.representation == "channel_static"
        return apply_fp8_linear(
            x,
            self.weight,
            self.weight_scale,
            input_scale=self.input_scale,
            cutlass_fp8_supported=use_channel_cutlass,
            pad_output=False,
        )

    def forward_prequantized(
        self,
        qinput: torch.Tensor,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run tensor-static GEMM from a unit-scale E4M3 activation.

        ``qinput`` must be the exact result of rounding the logical activation
        to ``output_dtype`` and then applying static E4M3 quantization with
        scale 1.0. The K3 AttnRes dual-output epilogue provides that contract.
        Channel-static weights retain the established ``forward`` path because
        their CUTLASS runner requires a repeated per-row activation scale.
        """
        if not self.supports_prequantized_static_input:
            raise RuntimeError(
                "K3 prequantized activation handoff requires tensor_static "
                f"weights, got {self.representation!r}."
            )
        if qinput.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "K3 prequantized activation must be float8_e4m3fn, "
                f"got {qinput.dtype}."
            )
        if qinput.ndim < 2 or qinput.shape[-1] != self.weight.shape[0]:
            raise ValueError(
                "K3 prequantized activation has incompatible shape: "
                f"input={tuple(qinput.shape)}, weight={tuple(self.weight.shape)}."
            )
        if not qinput.is_contiguous():
            raise ValueError("K3 prequantized activation must be contiguous.")
        if output_dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(
                "K3 prequantized output dtype must be BF16/FP16, "
                f"got {output_dtype}."
            )
        output_shape = (*qinput.shape[:-1], self.weight.shape[1])
        if qinput.numel() == 0:
            return qinput.new_empty(output_shape, dtype=output_dtype)

        qinput_2d = qinput.view(-1, qinput.shape[-1])
        output = torch._scaled_mm(
            qinput_2d,
            self.weight,
            scale_a=self.input_scale,
            scale_b=self.weight_scale,
            out_dtype=output_dtype,
        )
        # Accommodate PyTorch builds whose scaled_mm binding retains the
        # historical (output, amax) return without changing the hot path.
        if isinstance(output, tuple):
            output = output[0]
        return output.view(*output_shape)

    def _save_to_state_dict(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "K3 target dense FP8 is an online runtime representation and "
            "cannot be saved/reloaded as a sharded state checkpoint. Reload "
            "the original BF16 checkpoint and repeat the post-load conversion."
        )


class K3TargetFP8State:
    """Owns conversion accounting and per-linear source pools for one model."""

    def __init__(self, scope: str, representation: str) -> None:
        scope = scope.strip().lower()
        representation = representation.strip().lower()
        if scope not in _SUPPORTED_SCOPES:
            raise ValueError(
                f"Unsupported SGLANG_K3_TARGET_DENSE_FP8={scope!r}; "
                f"expected one of {sorted(_SUPPORTED_SCOPES)}."
            )
        if representation not in _SUPPORTED_REPRESENTATIONS:
            raise ValueError(
                "Unsupported SGLANG_K3_TARGET_DENSE_FP8_REPRESENTATION="
                f"{representation!r}; expected one of "
                f"{sorted(_SUPPORTED_REPRESENTATIONS)}."
            )
        self.scope = scope
        self.representation = representation
        self.stats = K3TargetFP8MemoryStats(
            scope=scope,
            representation=representation,
        )
        self._source_groups: list[K3TargetFP8SourceGroup] = []
        self._created_ids_by_role: defaultdict[str, set[int]] = defaultdict(set)
        self._converted_ids_by_role: defaultdict[str, set[int]] = defaultdict(set)
        self._expected_ids_by_role: dict[str, set[int]] | None = None
        self._configured_kda_count: int | None = None
        self._range_diagnostics: list[dict[str, object]] = []
        self._loaded_checkpoint_components: defaultdict[tuple[str, int], set[str]] = (
            defaultdict(set)
        )
        self._checkpoint_load_started = False
        self._checkpoint_load_finished = False
        self._released_groups = 0
        self._finalized = False
        self._conversion_started = False
        self._memory_before: dict[str, int] | None = None

    @classmethod
    def from_env(cls) -> K3TargetFP8State:
        return cls(
            envs.SGLANG_K3_TARGET_DENSE_FP8.get(),
            envs.SGLANG_K3_TARGET_DENSE_FP8_REPRESENTATION.get(),
        )

    @property
    def enabled(self) -> bool:
        return self.scope != "off"

    @property
    def range_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(self._range_diagnostics)

    def role_enabled(self, role: str) -> bool:
        if role == _FRONT_ROLE:
            return self.scope in ("front", "wide")
        if role == _KDA_QKVG_ROLE:
            return self.scope == "wide"
        return False

    def configure_expected_layer_ids(
        self,
        *,
        local_front_ids: Sequence[int],
        local_kda_ids: Sequence[int],
        configured_kda_count: int,
    ) -> None:
        """Validate construction against config/PP-derived local layer IDs."""
        expected: dict[str, set[int]] = {}
        if self.role_enabled(_FRONT_ROLE):
            expected[_FRONT_ROLE] = set(local_front_ids)
        if self.role_enabled(_KDA_QKVG_ROLE):
            expected[_KDA_QKVG_ROLE] = set(local_kda_ids)
        self._expected_ids_by_role = expected
        self._configured_kda_count = configured_kda_count

        mismatches = {
            role: {
                "expected": sorted(expected_ids),
                "created": sorted(self._created_ids_by_role.get(role, set())),
            }
            for role, expected_ids in expected.items()
            if self._created_ids_by_role.get(role, set()) != expected_ids
        }
        if mismatches:
            raise RuntimeError(
                "K3 target FP8 construction does not match model config: "
                f"expected/created IDs={mismatches}, configured_global_kda="
                f"{configured_kda_count}. Wide requires the already-merged "
                "plain-TP KDA qkvg path."
            )

    def begin_checkpoint_load(self) -> None:
        """Enable completeness tracking for the standard streaming loader."""
        if not self.enabled:
            return
        if self._checkpoint_load_started:
            raise RuntimeError(
                "K3 target FP8 checkpoint load tracking started more than once."
            )
        if self._finalized:
            raise RuntimeError(
                "K3 target FP8 cannot load a checkpoint after conversion."
            )
        self._checkpoint_load_started = True

    def mark_checkpoint_component(
        self,
        role: str,
        identifier: int,
        component: str,
    ) -> None:
        if not self.role_enabled(role):
            return
        if not self._checkpoint_load_started or self._checkpoint_load_finished:
            raise RuntimeError(
                "K3 target FP8 checkpoint component was recorded outside "
                "the streaming load window."
            )
        supported = _CHECKPOINT_COMPONENTS_BY_ROLE[role]
        if component not in supported:
            raise ValueError(
                f"Unsupported K3 target FP8 checkpoint component "
                f"{role}/{component}; expected one of {sorted(supported)}."
            )
        if (
            self._expected_ids_by_role is not None
            and identifier not in self._expected_ids_by_role.get(role, set())
        ):
            raise RuntimeError(
                "K3 target FP8 loaded a selected component for an unexpected "
                f"local layer: {role} {identifier}/{component}."
            )
        key = (role, identifier)
        if component in self._loaded_checkpoint_components[key]:
            raise RuntimeError(
                "K3 target FP8 checkpoint contains a duplicate selected "
                f"component: {role} {identifier}/{component}."
            )
        self._loaded_checkpoint_components[key].add(component)

    def finish_checkpoint_load(self) -> None:
        """Reject missing selected shards before any online quantization."""
        if not self.enabled:
            return
        if not self._checkpoint_load_started or self._checkpoint_load_finished:
            raise RuntimeError(
                "K3 target FP8 checkpoint load tracking has an invalid lifetime."
            )
        if self._expected_ids_by_role is None:
            raise RuntimeError(
                "K3 target FP8 checkpoint load finished before model topology "
                "was configured."
            )
        mismatches = {}
        for role, expected_ids in self._expected_ids_by_role.items():
            expected_components = _CHECKPOINT_COMPONENTS_BY_ROLE[role]
            for identifier in expected_ids:
                loaded = self._loaded_checkpoint_components.get(
                    (role, identifier), set()
                )
                if loaded != expected_components:
                    mismatches[f"{role}:{identifier}"] = {
                        "missing": sorted(expected_components - loaded),
                        "unexpected": sorted(loaded - expected_components),
                    }
        if mismatches:
            raise RuntimeError(
                "K3 target FP8 checkpoint is incomplete for selected dense "
                f"weights: {mismatches}."
            )
        self._checkpoint_load_finished = True

    def new_source_group(
        self,
        role: str,
        identifier: int,
    ) -> K3TargetFP8SourceGroup | None:
        if not self.role_enabled(role) or not torch.cuda.is_available():
            return None
        if self._finalized:
            raise RuntimeError("K3 target FP8 source pool is already finalized.")
        if identifier in self._created_ids_by_role[role]:
            raise RuntimeError(
                f"Duplicate K3 target FP8 source group for {role} {identifier}."
            )
        group = K3TargetFP8SourceGroup(
            owner=self,
            role=role,
            identifier=identifier,
            pool=torch.cuda.MemPool(use_on_oom=False),
        )
        self._source_groups.append(group)
        self._created_ids_by_role[role].add(identifier)
        return group

    def convert(
        self,
        source: torch.Tensor,
        role: str,
        identifier: int,
        *,
        component_names: Sequence[str] | None = None,
        component_rows: Sequence[int] | None = None,
    ) -> K3TargetFP8Linear:
        if not self.role_enabled(role):
            raise RuntimeError(f"K3 target FP8 role {role!r} is disabled.")
        if identifier in self._converted_ids_by_role[role]:
            raise RuntimeError(
                f"Duplicate K3 target FP8 conversion for {role} {identifier}."
            )
        if not source.is_cuda:
            raise RuntimeError(
                "K3 target FP8 cannot convert a CPU/offloaded source for "
                f"{role} {identifier}."
            )
        self.begin_conversion()
        source_bytes = source.numel() * source.element_size()

        collect_ranges = envs.SGLANG_K3_TARGET_DENSE_FP8_RANGE_DIAGNOSTICS.get()
        source_range: dict[str, object] | None = None
        if collect_ranges:
            source_range = self._source_range_diagnostic(
                source,
                role,
                component_names=component_names,
                component_rows=component_rows,
            )

        converted = K3TargetFP8Linear.from_bf16(
            source,
            representation=self.representation,
        )
        self.stats.replaced_source_bytes += source_bytes
        self.stats.final_weight_bytes += (
            converted.weight.numel() * converted.weight.element_size()
        )
        self.stats.final_scale_bytes += converted.resident_scale_bytes
        self.stats.replaced_linears += 1
        self._converted_ids_by_role[role].add(identifier)

        if source_range is not None:
            scale = converted.weight_scale.detach().float()
            logical_weight = converted.logical_weight.detach()
            fp8_max = torch.finfo(logical_weight.dtype).max
            source_range.update(
                {
                    "weight_scale_min": scale.min().item(),
                    "weight_scale_max": scale.max().item(),
                    "identifier": identifier,
                    "saturation_fraction": (logical_weight.abs() == fp8_max)
                    .float()
                    .mean()
                    .item(),
                }
            )
            self._range_diagnostics.append(source_range)
            rank0_log(f"K3 target dense FP8 range: {source_range}")
        return converted

    def _source_range_diagnostic(
        self,
        source: torch.Tensor,
        role: str,
        *,
        component_names: Sequence[str] | None,
        component_rows: Sequence[int] | None,
    ) -> dict[str, object]:
        if (component_names is None) != (component_rows is None):
            raise ValueError(
                "K3 target FP8 diagnostics require both component names and rows."
            )
        components: list[dict[str, int | float | str]] = []
        if component_names is not None and component_rows is not None:
            if len(component_names) != len(component_rows):
                raise ValueError(
                    "K3 target FP8 diagnostic component counts do not match."
                )
            if sum(component_rows) != source.shape[0]:
                raise ValueError(
                    "K3 target FP8 diagnostic rows do not cover the source."
                )
            off = 0
            for name, rows in zip(component_names, component_rows):
                value = source[off : off + rows]
                minimum, maximum = value.aminmax()
                components.append(
                    {
                        "name": name,
                        "rows": rows,
                        "min": minimum.item(),
                        "max": maximum.item(),
                        "absmax": max(
                            abs(minimum.item()),
                            abs(maximum.item()),
                        ),
                    }
                )
                off += rows

        minimum, maximum = source.aminmax()
        return {
            "role": role,
            "representation": self.representation,
            "source_min": minimum.item(),
            "source_max": maximum.item(),
            "source_absmax": max(abs(minimum.item()), abs(maximum.item())),
            "components": components,
        }

    def begin_conversion(self) -> None:
        """Start the conversion-only CUDA-memory measurement window."""
        if self._conversion_started:
            return
        self._conversion_started = True
        if (
            not self.enabled
            or not torch.cuda.is_available()
            or not envs.SGLANG_K3_TARGET_DENSE_FP8_MEMORY_DIAGNOSTICS.get()
        ):
            return
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self._memory_before = _cuda_memory_snapshot()

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        live_groups = [group for group in self._source_groups if not group.released]
        if live_groups:
            raise RuntimeError(
                "K3 target FP8 reached finalization with "
                f"{len(live_groups)} live BF16 source pools. A skipped "
                "conversion or source alias would defeat the capacity win."
            )
        if self._expected_ids_by_role is not None:
            mismatches = {
                role: {
                    "expected": sorted(expected_ids),
                    "converted": sorted(self._converted_ids_by_role.get(role, set())),
                }
                for role, expected_ids in self._expected_ids_by_role.items()
                if self._converted_ids_by_role.get(role, set()) != expected_ids
            }
            if mismatches:
                raise RuntimeError(
                    "K3 target FP8 conversion IDs do not match model config: "
                    f"expected/converted={mismatches}."
                )
        self._source_groups.clear()
        if self.enabled and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if self.enabled:
            stats = self.stats.as_dict()
            rank0_log(
                "K3 target dense FP8 replacement: "
                + ", ".join(f"{key}={value}" for key, value in stats.items())
                + ", created_ids_by_role="
                + str(
                    {
                        role: sorted(ids)
                        for role, ids in self._created_ids_by_role.items()
                    }
                )
                + ", converted_ids_by_role="
                + str(
                    {
                        role: sorted(ids)
                        for role, ids in self._converted_ids_by_role.items()
                    }
                )
                + f", configured_global_kda={self._configured_kda_count}"
            )
            if self._memory_before is not None:
                after = _cuda_memory_snapshot()
                before = self._memory_before
                rank0_log(
                    "K3 target dense FP8 CUDA memory: "
                    + ", ".join(
                        f"{key}_before={before[key]}, "
                        f"{key}_after={after[key]}, "
                        f"{key}_delta={after[key] - before[key]:+d}"
                        for key in before
                    )
                )

    def _on_source_group_released(self) -> None:
        self._released_groups += 1
        if self._released_groups % _EMPTY_DEFAULT_CACHE_EVERY_GROUPS == 0:
            torch.cuda.empty_cache()


class K3TargetFP8SourceGroup:
    """One conversion unit whose private BF16 source HBM is returned immediately."""

    def __init__(
        self,
        *,
        owner: K3TargetFP8State,
        role: str,
        identifier: int,
        pool: torch.cuda.MemPool,
    ) -> None:
        self.owner = owner
        self.role = role
        self.identifier = identifier
        self._pool: torch.cuda.MemPool | None = pool
        self._staged_storage_ptrs: dict[int, int] = {}
        self.released = False

    def allocation(self):
        if self._pool is None or self.released:
            raise RuntimeError("K3 target FP8 source group is already released.")
        return torch.cuda.use_mem_pool(self._pool)

    def stage_linear_weight(self, module: nn.Module) -> None:
        """Move an uninitialized checkpoint destination into this private pool."""
        weight = getattr(module, "weight", None)
        if not isinstance(weight, nn.Parameter):
            raise TypeError(f"{type(module).__name__} has no weight Parameter.")
        if weight.device.type != "cuda":
            raise RuntimeError(
                "K3 target FP8 requires CUDA-resident checkpoint "
                f"destinations, got {weight.device} for "
                f"{self.role} {self.identifier}. CPU offload is unsupported."
            )
        if id(weight) in self._staged_storage_ptrs:
            raise RuntimeError(
                "K3 target FP8 attempted to stage the same Parameter twice "
                f"for {self.role} {self.identifier}."
            )
        with self.allocation():
            staged = torch.empty(
                weight.shape,
                dtype=weight.dtype,
                device=weight.device,
                memory_format=torch.contiguous_format,
            )
        weight.data = staged
        weight.requires_grad_(False)
        self._staged_storage_ptrs[id(weight)] = weight.untyped_storage().data_ptr()
        self.owner.stats.staged_source_bytes += staged.numel() * staged.element_size()

    def validate_staged_linear_weight(self, module: nn.Module) -> None:
        """Ensure loading preserved the destination allocated in this pool."""
        weight = getattr(module, "weight", None)
        if not isinstance(weight, nn.Parameter):
            raise TypeError(f"{type(module).__name__} has no weight Parameter.")
        expected_ptr = self._staged_storage_ptrs.get(id(weight))
        actual_ptr = weight.untyped_storage().data_ptr()
        if expected_ptr is None or actual_ptr != expected_ptr:
            raise RuntimeError(
                "K3 target FP8 checkpoint loading rebound a staged weight "
                f"outside its private source pool for {self.role} "
                f"{self.identifier}: expected_ptr={expected_ptr}, "
                f"actual_ptr={actual_ptr}. The no-expandable-segments HBM "
                "recovery guarantee would no longer hold."
            )

    def release(self) -> None:
        if self.released:
            return
        if self._pool is None:
            raise RuntimeError("K3 target FP8 source group lost its pool.")
        gc.collect()
        torch.cuda.synchronize()
        use_count = self._pool.use_count()
        if use_count != 1:
            raise RuntimeError(
                "K3 target FP8 source pool still has live users after "
                f"weight replacement (use_count={use_count}). A BF16 "
                "source alias would defeat the capacity win."
            )
        self._pool = None
        self.released = True
        gc.collect()
        self.owner._on_source_group_released()


def moe_front_role() -> str:
    return _FRONT_ROLE


def kda_qkvg_role() -> str:
    return _KDA_QKVG_ROLE
