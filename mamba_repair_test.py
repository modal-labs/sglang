"""Dependency-light checks for the mamba-only write-through repair op.

Covers the bug where a hybrid-SSM node whose host mamba checkpoint was
evicted (while its Full-KV host backup survived) became permanently
L2-unreachable: the write-through trigger's done-check (node.backuped) is
KV-only and the write op always shipped KV+mamba together.

The production modules are intentionally not imported (an SGLang import
pulls GPU deps); real method bodies are extracted with ``ast`` and executed
against small fakes, in the style of r1_reconcile_test.py.

Run: ``python3 mamba_repair_test.py`` (torch required).
"""

from __future__ import annotations

import ast
import enum
import logging
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parent
CACHE_PATH = REPO / "python/sglang/srt/mem_cache/unified_radix_cache.py"
MAMBA_PATH = REPO / "python/sglang/srt/mem_cache/unified_cache_components/mamba_component.py"

PASSED = 0


def check(condition, message):
    global PASSED
    if not condition:
        raise AssertionError(message)
    PASSED += 1


def _method_node(path: Path, class_name: str, method_name: str):
    source = path.read_text()
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    return source, method


def _extract_methods(path: Path, class_name: str, names, namespace):
    result = {}
    for name in names:
        source, node = _method_node(path, class_name, name)
        segment = textwrap.dedent(ast.get_source_segment(source, node))
        local_ns = dict(namespace)
        exec(
            compile("from __future__ import annotations\n" + segment, str(path), "exec"),
            local_ns,
        )
        result[name] = local_ns[name]
    return result


# ---- Shared fakes -----------------------------------------------------------


class ComponentType(enum.IntEnum):
    FULL = 0
    MAMBA = 1
    SWA = 2


class EvictLayer(enum.IntFlag):
    DEVICE = 1
    HOST = 2
    ALL = DEVICE | HOST


class PoolName(enum.Enum):
    KV = "kv"
    MAMBA = "mamba"
    SWA = "swa"


class CacheTransferPhase(enum.Enum):
    BACKUP_HOST = 1
    LOAD_BACK = 2
    BACKUP_STORAGE = 3
    PREFETCH = 4


class PoolTransfer:
    def __init__(self, name, host_indices=None, device_indices=None, keys=None,
                 hit_policy=None, nodes_to_load=None, indices_from_pool=None):
        self.name = name
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.keys = keys
        self.hit_policy = hit_policy
        self.nodes_to_load = nodes_to_load
        self.indices_from_pool = indices_from_pool


BASE_COMPONENT_TYPE = ComponentType.FULL
_HICACHE_OP_WRITE = 1
_HICACHE_OP_WRITE_MAMBA_REPAIR = 7


def make_component_data():
    return SimpleNamespace(value=None, host_value=None, lock_ref=0, host_lock_ref=0)


class FakeNode:
    counter = 1000

    def __init__(self, parent=None):
        self.component_data = [make_component_data() for _ in range(3)]
        self.parent = parent
        self.children = {}
        self.hit_count = 0
        self.write_through_pending_id = None
        self.id = FakeNode.counter
        FakeNode.counter += 1

    @property
    def backuped(self):
        return self.component_data[ComponentType.FULL].host_value is not None

    @property
    def evicted(self):
        return (
            self.parent is not None
            and self.component_data[ComponentType.FULL].value is None
        )


class FakeLRU:
    def __init__(self):
        self.nodes = set()
        self.removed = []

    def in_list(self, node):
        return node in self.nodes

    def remove_node(self, node):
        self.nodes.discard(node)
        self.removed.append(node)

    def insert_mru(self, node):
        self.nodes.add(node)


NAMESPACE = {
    "ComponentType": ComponentType,
    "EvictLayer": EvictLayer,
    "PoolName": PoolName,
    "PoolTransfer": PoolTransfer,
    "CacheTransferPhase": CacheTransferPhase,
    "BASE_COMPONENT_TYPE": BASE_COMPONENT_TYPE,
    "_HICACHE_OP_WRITE": _HICACHE_OP_WRITE,
    "_HICACHE_OP_WRITE_MAMBA_REPAIR": _HICACHE_OP_WRITE_MAMBA_REPAIR,
    "torch": torch,
    "logger": logging.getLogger("mamba_repair_test"),
    "Optional": None,  # annotations are strings under future-import
    "UnifiedTreeNode": object,
}

CACHE_METHODS = _extract_methods(
    CACHE_PATH,
    "UnifiedRadixCache",
    [
        "_mamba_backup_missing",
        "_maybe_repair_mamba_backup",
        "_inc_hit_count",
        "write_backup",
    ],
    NAMESPACE,
)

MAMBA_METHODS = _extract_methods(
    MAMBA_PATH,
    "MambaComponent",
    ["evict_component"],
    NAMESPACE,
)


def make_cache(*, write_policy="write_through", threshold=1):
    cache = SimpleNamespace()
    cache.root_node = FakeNode()
    cache.write_through_threshold = threshold
    cache.tp_world_size = 1
    cache.pp_size = 1
    cache.device = torch.device("cpu")
    cache.ongoing_write_through = {}
    cache.write_backup_calls = []

    mamba_comp = SimpleNamespace(component_type=ComponentType.MAMBA)
    full_comp = SimpleNamespace(component_type=ComponentType.FULL)
    cache.components = {
        ComponentType.FULL: full_comp,
        ComponentType.MAMBA: mamba_comp,
    }
    cache._components_tuple = (full_comp, mamba_comp)

    cache.cache_controller = SimpleNamespace(write_policy=write_policy)

    for name in (
        "_mamba_backup_missing",
        "_maybe_repair_mamba_backup",
        "_inc_hit_count",
    ):
        setattr(cache, name, types.MethodType(CACHE_METHODS[name], cache))
    return cache


def make_node(cache, *, kv_backuped, mamba_device, mamba_host, pending=None):
    node = FakeNode(parent=cache.root_node)
    node.component_data[ComponentType.FULL].value = torch.tensor([10, 11, 12])
    if kv_backuped:
        node.component_data[ComponentType.FULL].host_value = torch.tensor([1, 2, 3])
    if mamba_device:
        node.component_data[ComponentType.MAMBA].value = torch.tensor([5])
    if mamba_host:
        node.component_data[ComponentType.MAMBA].host_value = torch.tensor([7])
    node.write_through_pending_id = pending
    return node


# ---- (a) predicate ----------------------------------------------------------


def test_predicate_truth_table():
    cache = make_cache()
    # Repair case: KV backuped, device mamba present, host mamba missing.
    node = make_node(cache, kv_backuped=True, mamba_device=True, mamba_host=False)
    check(cache._mamba_backup_missing(node), "predicate: repair case must be True")

    # Host mamba present -> no repair.
    node = make_node(cache, kv_backuped=True, mamba_device=True, mamba_host=True)
    check(not cache._mamba_backup_missing(node), "predicate: host-resident must be False")

    # KV not backuped -> normal write-through owns this node.
    node = make_node(cache, kv_backuped=False, mamba_device=True, mamba_host=False)
    check(not cache._mamba_backup_missing(node), "predicate: unbackuped KV must be False")

    # No device mamba state to re-ship.
    node = make_node(cache, kv_backuped=True, mamba_device=False, mamba_host=False)
    check(not cache._mamba_backup_missing(node), "predicate: no device state must be False")

    # Device-evicted node (nothing to read for the D2H copy).
    node = make_node(cache, kv_backuped=True, mamba_device=True, mamba_host=False)
    node.component_data[ComponentType.FULL].value = None
    check(not cache._mamba_backup_missing(node), "predicate: evicted node must be False")

    # Root node excluded.
    check(not cache._mamba_backup_missing(cache.root_node), "predicate: root must be False")

    # In-flight write-through op excluded (bounded: one repair per node).
    node = make_node(
        cache, kv_backuped=True, mamba_device=True, mamba_host=False, pending=42
    )
    check(not cache._mamba_backup_missing(node), "predicate: pending op must be False")

    # No MAMBA component configured -> never fires.
    cache_no_mamba = make_cache()
    del cache_no_mamba.components[ComponentType.MAMBA]
    node = make_node(cache_no_mamba, kv_backuped=True, mamba_device=True, mamba_host=False)
    check(
        not cache_no_mamba._mamba_backup_missing(node),
        "predicate: no mamba component must be False",
    )


# ---- (b) eviction clears the marker ----------------------------------------


def test_host_eviction_clears_marker():
    host_lru = FakeLRU()
    freed = []
    comp = SimpleNamespace(
        component_type=ComponentType.MAMBA,
        _mamba_pool_host=SimpleNamespace(free=lambda idx: freed.append(idx)),
        _free_mamba_value=lambda v: None,
        cache=SimpleNamespace(
            component_evictable_size_={ComponentType.MAMBA: 10},
            host_lru_lists={ComponentType.MAMBA: host_lru},
            lru_lists={ComponentType.MAMBA: FakeLRU()},
        ),
    )
    evict_component = types.MethodType(MAMBA_METHODS["evict_component"], comp)

    node = FakeNode()
    host_value = torch.tensor([7])
    node.component_data[ComponentType.MAMBA].host_value = host_value
    host_lru.insert_mru(node)

    _, host_freed = evict_component(node, target=EvictLayer.HOST)
    cd = node.component_data[ComponentType.MAMBA]
    check(cd.host_value is None, "host eviction must clear the residency marker")
    check(host_freed == 1, "host eviction must report freed slot count")
    check(len(freed) == 1 and torch.equal(freed[0], host_value),
          "host eviction must return the slot to the host pool")
    check(not host_lru.in_list(node), "host eviction must detach node from host LRU")

    # The cleared marker makes the repair predicate observable.
    cache = make_cache()
    node2 = make_node(cache, kv_backuped=True, mamba_device=True, mamba_host=True)
    comp2 = SimpleNamespace(
        component_type=ComponentType.MAMBA,
        _mamba_pool_host=SimpleNamespace(free=lambda idx: None),
        _free_mamba_value=lambda v: None,
        cache=SimpleNamespace(
            component_evictable_size_={ComponentType.MAMBA: 10},
            host_lru_lists={ComponentType.MAMBA: FakeLRU()},
            lru_lists={ComponentType.MAMBA: FakeLRU()},
        ),
    )
    check(not cache._mamba_backup_missing(node2), "pre-eviction: predicate False")
    types.MethodType(MAMBA_METHODS["evict_component"], comp2)(
        node2, target=EvictLayer.HOST
    )
    check(cache._mamba_backup_missing(node2),
          "post-eviction: predicate must flip True (repair reachable)")


# ---- (c) mamba-only op shape ------------------------------------------------


class FakeController:
    def __init__(self):
        self.write_policy = "write_through"
        self.mem_pool_host = SimpleNamespace(available_size=lambda: 1 << 20)
        self.committed_ops = []
        self.aborted = []

    def reserve_write(self, device_indices, node_id=-1, extra_pools=None, *,
                      allow_evict=False, priority=None):
        # Mirror _reserve_pool_transfers: allocate host slots per extra pool.
        for pool in extra_pools or []:
            if pool.host_indices is None and pool.device_indices is not None:
                pool.host_indices = torch.tensor([7] * len(pool.device_indices))
        return SimpleNamespace(
            host_indices=torch.empty(
                (len(device_indices),), dtype=torch.int64
            ),
            device_indices=device_indices,
            node_id=node_id,
            extra_pools=list(extra_pools or []),
        )

    def commit_write(self, reservation):
        self.committed_ops.append(reservation)
        return reservation.host_indices

    def abort_write(self, reservation):
        self.aborted.append(reservation)


class RecordingComponent:
    def __init__(self, component_type):
        self.component_type = component_type
        self.built = []
        self.commits = []

    def build_hicache_transfers(self, node, phase, **kwargs):
        self.built.append((node, phase))
        cd = node.component_data[self.component_type]
        if cd.value is None:
            return None
        return [PoolTransfer(name=PoolName.MAMBA, device_indices=cd.value)]

    def commit_hicache_transfer(self, node, phase, transfers=(), **kwargs):
        self.commits.append((node, phase, transfers))
        if self.component_type == ComponentType.MAMBA:
            cd = node.component_data[self.component_type]
            if transfers and transfers[0].host_indices is not None:
                if cd.host_value is None:
                    cd.host_value = transfers[0].host_indices.clone()


def make_write_backup_cache():
    cache = make_cache()
    controller = FakeController()
    cache.cache_controller = controller

    full_comp = RecordingComponent(ComponentType.FULL)
    mamba_comp = RecordingComponent(ComponentType.MAMBA)
    cache.components = {
        ComponentType.FULL: full_comp,
        ComponentType.MAMBA: mamba_comp,
    }
    cache._components_tuple = (full_comp, mamba_comp)

    cache.fence_state_read = lambda: None
    cache._watermark_evict_host_pools = lambda: None
    cache._build_sidecar_transfers = lambda phase, kv, comps: []
    cache.evict_host = lambda n, component_type=None: n
    cache.inc_lock_ref = lambda node: SimpleNamespace(
        to_dec_params=lambda: SimpleNamespace()
    )
    cache._all_ranks_succeeded = lambda *a, **k: True

    def _track(node, lock_params):
        node.write_through_pending_id = node.id
        cache.ongoing_write_through[node.id] = (node, lock_params, [node])

    cache._track_write_through_node = _track
    cache.write_backup = types.MethodType(CACHE_METHODS["write_backup"], cache)
    return cache, controller, full_comp, mamba_comp


def test_mamba_only_op_shape():
    cache, controller, full_comp, mamba_comp = make_write_backup_cache()
    node = make_node(cache, kv_backuped=True, mamba_device=True, mamba_host=False)

    written = cache.write_backup(node, mamba_only=True)
    check(written == 1, f"mamba-only success must report 1 state, got {written}")

    check(len(controller.committed_ops) == 1, "exactly one op must be committed")
    op = controller.committed_ops[0]
    check(len(op.device_indices) == 0, "op KV payload must be empty")
    check(len(op.host_indices) == 0, "op must allocate zero KV host slots")
    mamba_pools = [p for p in op.extra_pools if p.name == PoolName.MAMBA]
    check(len(mamba_pools) == 1, "op must carry exactly one mamba transfer")
    check(
        mamba_pools[0].device_indices is not None
        and len(mamba_pools[0].device_indices) == 1,
        "mamba transfer must carry the device state",
    )

    # FULL commit skipped: an empty-KV commit would clobber the existing
    # host_value (FullComponent.commit_hicache_transfer overwrites).
    check(not full_comp.commits, "FULL commit must be skipped for mamba-only op")
    check(len(mamba_comp.commits) == 1, "mamba commit must run")
    cd = node.component_data[ComponentType.MAMBA]
    check(cd.host_value is not None, "mamba host residency must be published")
    fd = node.component_data[ComponentType.FULL]
    check(
        fd.host_value is not None and len(fd.host_value) == 3,
        "existing KV host backup must be untouched",
    )
    check(node.write_through_pending_id == node.id,
          "op must be tracked in the write-through ack machinery")
    check(node.id in cache.ongoing_write_through, "ongoing op must be registered")


def test_mamba_only_guards():
    cache, controller, _, _ = make_write_backup_cache()

    # Not backuped -> refuse (normal write-through owns it).
    node = make_node(cache, kv_backuped=False, mamba_device=True, mamba_host=False)
    check(cache.write_backup(node, mamba_only=True) == 0,
          "mamba-only on unbackuped node must return 0")

    # No device mamba state -> refuse.
    node = make_node(cache, kv_backuped=True, mamba_device=False, mamba_host=False)
    check(cache.write_backup(node, mamba_only=True) == 0,
          "mamba-only without device state must return 0")

    # Pending op -> refuse (no double-scheduling).
    node = make_node(cache, kv_backuped=True, mamba_device=True, mamba_host=False,
                     pending=99)
    check(cache.write_backup(node, mamba_only=True) == 0,
          "mamba-only with in-flight op must return 0")
    check(not controller.committed_ops, "guard failures must not commit ops")


# ---- (d) trigger: bounded, not re-fired for in-flight ops -------------------


def test_trigger_fires_once_and_not_while_in_flight():
    cache = make_cache(threshold=1)
    calls = []

    def fake_write_backup(node, write_back=False, mamba_only=False):
        calls.append((node.id, mamba_only))
        # Mirror the real op: track in-flight so re-triggers are suppressed.
        node.write_through_pending_id = node.id
        cache.ongoing_write_through[node.id] = (node, None, [node])
        return 1

    cache.write_backup = fake_write_backup

    node = make_node(cache, kv_backuped=True, mamba_device=True, mamba_host=False)
    node.hit_count = 1

    cache._maybe_repair_mamba_backup(node)
    check(calls == [(node.id, True)], "repair must fire exactly once")

    # Second pass while the op is still in flight: no re-fire.
    cache._maybe_repair_mamba_backup(node)
    check(len(calls) == 1, "repair must not re-fire for an in-flight op")

    # Op acked and host_value published: predicate false, still no re-fire.
    node.write_through_pending_id = None
    node.component_data[ComponentType.MAMBA].host_value = torch.tensor([7])
    cache._maybe_repair_mamba_backup(node)
    check(len(calls) == 1, "repair must not re-fire once host-resident")


def test_inc_hit_count_routes_repair():
    cache = make_cache(threshold=1)
    calls = []
    cache.write_backup = lambda node, write_back=False, mamba_only=False: (
        calls.append((node.id, mamba_only)),
        node.__setattr__("write_through_pending_id", node.id),
        1,
    )[-1]

    # Backuped node with lost host mamba: walk hit triggers the repair.
    node = make_node(cache, kv_backuped=True, mamba_device=True, mamba_host=False)
    cache._inc_hit_count(node)
    check(calls == [(node.id, True)], "_inc_hit_count must route to mamba-only repair")

    # Unbackuped node: the normal (full) write-through fires, not the repair.
    calls.clear()
    node = make_node(cache, kv_backuped=False, mamba_device=True, mamba_host=False)
    cache._inc_hit_count(node)
    check(calls == [(node.id, False)], "unbackuped node must take the full backup path")

    # Chunked inserts and write_back policy never repair.
    calls.clear()
    node = make_node(cache, kv_backuped=True, mamba_device=True, mamba_host=False)
    cache._inc_hit_count(node, chunked=True)
    check(not calls, "chunked insert must not trigger repair")
    wb_cache = make_cache(write_policy="write_back")
    wb_calls = []
    wb_cache.write_backup = lambda *a, **k: wb_calls.append(1)
    node = make_node(wb_cache, kv_backuped=True, mamba_device=True, mamba_host=False)
    wb_cache._inc_hit_count(node)
    check(not wb_calls, "write_back policy must not trigger repair")


# ---- structural (AST) assertions --------------------------------------------


def test_ast_wiring():
    source, wb = _method_node(CACHE_PATH, "UnifiedRadixCache", "write_backup")
    names = {n.id for n in ast.walk(wb) if isinstance(n, ast.Name)}
    check("_HICACHE_OP_WRITE_MAMBA_REPAIR" in names,
          "write_backup consensus must use the distinct repair opcode")

    _, insert = _method_node(CACHE_PATH, "UnifiedRadixCache", "_insert_helper")
    repair_calls = [
        n
        for n in ast.walk(insert)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_maybe_repair_mamba_backup"
    ]
    check(len(repair_calls) == 1,
          "_insert_helper must schedule the repair for reused target nodes")

    # No new collective shapes: write_backup must not call _all_reduce directly.
    reduce_calls = [
        n
        for n in ast.walk(wb)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("_all_reduce", "all_reduce")
    ]
    check(not reduce_calls,
          "write_backup must only use the existing fingerprinted consensus")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests, {PASSED} checks passed")
