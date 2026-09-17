from types import SimpleNamespace
from unittest.mock import Mock

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


if __name__ == "__main__":
    import unittest

    unittest.main()
