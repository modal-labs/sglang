from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.log_utils import create_log_targets, log_json

_waypoint_loggers = None
_INTERNAL_REQUEST_PREFIXES = ("HEALTH_CHECK_",)


def request_waypoints_enabled() -> bool:
    try:
        return bool(get_global_server_args().enable_request_waypoint_logging)
    except Exception:
        return False


def _get_waypoint_loggers():
    global _waypoint_loggers
    if _waypoint_loggers is None:
        _waypoint_loggers = create_log_targets(targets=None, name_prefix=__name__)
    return _waypoint_loggers


def ms_from_s(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value * 1000.0, 3)


def count_mm_items(data: Any) -> int:
    if data is None:
        return 0
    if isinstance(data, list):
        return sum(count_mm_items(item) for item in data)
    return 1


def emit_request_waypoint(event: str, data: Dict[str, Any]) -> None:
    if not request_waypoints_enabled():
        return

    if data.get("no_logs"):
        return

    if _is_internal_request(data):
        return

    payload = {
        key: value
        for key, value in data.items()
        if key != "no_logs" and value is not None
    }
    log_json(_get_waypoint_loggers(), event, payload)


def sum_grid_patches(grid_values: Optional[Iterable[Any]]) -> int:
    if grid_values is None:
        return 0

    total = 0
    for grid in grid_values:
        if hasattr(grid, "tolist"):
            grid = grid.tolist()
        if isinstance(grid, list) and grid and isinstance(grid[0], list):
            total += sum(int(_prod_ints(item)) for item in grid)
        else:
            total += int(_prod_ints(grid))
    return total


def _prod_ints(values: Any) -> int:
    if values is None:
        return 0
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list):
        return int(values)

    out = 1
    for value in values:
        out *= int(value)
    return out


def _is_internal_request(data: Dict[str, Any]) -> bool:
    rid = data.get("rid")
    if isinstance(rid, str) and rid.startswith(_INTERNAL_REQUEST_PREFIXES):
        return True

    rids = data.get("rids")
    if isinstance(rids, list) and rids:
        return all(
            isinstance(item, str) and item.startswith(_INTERNAL_REQUEST_PREFIXES)
            for item in rids
        )

    return False
