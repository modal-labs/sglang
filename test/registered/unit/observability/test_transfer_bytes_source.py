"""Regression for mixed state-pool byte accounting; no GPU imports."""

from types import SimpleNamespace as NS

import pytest

from test_first_output_flush_source import SRT, load_node

PATH = SRT / "disaggregation/common/conn.py"
RECORD = load_node(PATH, "CommonKVSender", "_record_transfer_indices")
METRIC = load_node(PATH, "CommonKVSender", "get_transfer_metric")


@pytest.mark.parametrize("factor", [1, 4, 8])
def test_mixed_state_components_do_not_form_cross_products(factor):
    sender = NS(
        _transfer_num_kv_indices=0,
        _transfer_state_bytes=0,
        _transfer_metric=NS(transfer_total_bytes=0),
        kv_mgr=NS(
            kv_item_lens_sum=100,
            kv_args=NS(state_item_lens=[[3, 4], [10, 20]]),
            get_kv_replica_factor=lambda: factor,
        ),
    )
    RECORD(sender, range(8), [range(5), range(2)])
    assert METRIC(sender).transfer_total_bytes == (800 + 35 + 60) * factor
    # A subsequent chunk may transfer just one of the state components.
    RECORD(sender, range(2), [None, range(1)])
    assert METRIC(sender).transfer_total_bytes == (1000 + 35 + 90) * factor
    # No-state/empty chunks preserve accumulated state bytes.
    RECORD(sender, range(1), None)
    RECORD(sender, [], [])
    assert METRIC(sender).transfer_total_bytes == (1100 + 35 + 90) * factor
