from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSessionArrivalStamp(CustomTestCase):
    def test_session_turn_inherits_tokenizer_arrival_stamp(self):
        req = SimpleNamespace(
            finished_reason=None,
            to_finish=None,
            return_logprob=False,
            return_sampling_mask=False,
            return_routed_experts=False,
            routed_experts_start_len=0,
            origin_input_ids=[1],
            is_prefill_only=False,
            logprob_start_len=-1,
            extra_key=None,
            arrival_stamp=None,
            sampling_params=SimpleNamespace(top_k=1),
            time_stats=SimpleNamespace(
                trace_ctx=SimpleNamespace(abort=Mock()),
                set_quick_finish_time=Mock(),
                set_wait_queue_entry_time=Mock(),
            ),
        )
        session = SimpleNamespace(
            close_on_finish=False,
            create_req=Mock(return_value=req),
        )
        self_obj = SimpleNamespace(
            server_args=SimpleNamespace(
                enable_session_radix_cache=False,
                allow_auto_truncate=False,
                sampling_backend="torch",
            ),
            session_controller={"session-a": session},
            tokenizer=None,
            model_config=SimpleNamespace(
                vocab_size=16,
                hf_eos_token_id=[],
            ),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            disaggregation_mode=DisaggregationMode.NULL,
            spec_algorithm=SimpleNamespace(is_dflash_family=lambda: False),
            _maybe_namespace_elastic_radix_cache=Mock(),
            init_req_max_new_tokens=Mock(),
            _abort_on_queued_limit=Mock(return_value=False),
            _slo_admission_check=Mock(return_value=False),
            _prefetch_kvcache=Mock(),
            waiting_queue=[],
            _add_request_to_queue=Mock(),
            grammar_manager=SimpleNamespace(
                process_req_with_grammar=Mock(return_value=False)
            ),
            max_req_input_len=1024,
        )
        recv_req = SimpleNamespace(
            session_params=SimpleNamespace(id="session-a"),
            session_id=None,
            mm_inputs=None,
            return_logprob=False,
            return_routed_experts=False,
            logprob_start_len=-1,
            arrival_stamp=123.456,
        )

        Scheduler.handle_generate_request(self_obj, recv_req)

        self.assertEqual(req.arrival_stamp, 123.456)
        self_obj._add_request_to_queue.assert_called_once_with(req)

    def test_slo_admission_uses_arrival_stamp_not_rank_local_clock(self):
        """_slo_admission_check must base virtual_arrival on the tokenizer's
        broadcast arrival_stamp (same clock domain as r_vtime), not a
        rank-local perf_counter — otherwise every queued req looks like
        work-ahead and fresh admissions get spuriously 429'd."""
        send_output = Mock()
        scheduler = object.__new__(Scheduler)
        scheduler.server_args = SimpleNamespace(
            schedule_policy="openrouter_slo",
            slo_prefill_tokens_per_s=1000.0,
            slo_ttft_slope_ms_per_uncached_token=1.0,
            slo_ttft_base_s=0.0,
            slo_429_margin_s=0.0,
            openrouter_slo_429=True,
        )
        scheduler.chunked_req = None
        scheduler.tree_cache = Mock()
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=send_output)
        )

        req = SimpleNamespace(
            rid="incoming",
            origin_input_ids=[0] * 100,
            output_ids=[],
            finished_reason=None,
            arrival_stamp=1000.0,
            num_matched_prefix_tokens=0,
            min_uncached_seen=None,
            time_stats=SimpleNamespace(trace_ctx=SimpleNamespace(abort=Mock())),
        )
        # Queued req whose virtual arrival lands just AFTER the incoming
        # req's (1000.0 + 0.001*100 = 1000.1) -> not work ahead.
        queued = SimpleNamespace(
            rid="queued",
            origin_input_ids=[0] * 100,
            num_matched_prefix_tokens=0,
            min_uncached_seen=100,
            arrival_stamp=1000.0 + 0.001 * 100 + 1e-6,
        )
        scheduler.waiting_queue = [queued]

        # uncached=100 -> virtual_arrival=1000.1; only own work counted ->
        # predicted 0.1s <= limit 0.1s -> admitted. With a rank-local
        # perf_counter base (>> 1000.0) queued counts too -> 0.2s -> 429.
        with patch("sglang.srt.managers.scheduler.match_prefix_for_req", Mock()):
            self.assertFalse(scheduler._slo_admission_check(req))
        send_output.assert_not_called()


if __name__ == "__main__":
    import unittest

    unittest.main()
