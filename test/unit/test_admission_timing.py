import ast, types, os, unittest, importlib.util
from unittest.mock import MagicMock, patch
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "python/sglang/srt"


class Tests(unittest.TestCase):
    def test_parser(self):
        spec = importlib.util.spec_from_file_location(
            "timing", root / "observability/admission_timing.py"
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        for bad in [None, "bad", "nan", "inf", "-1"]:
            self.assertEqual(m.parse_admission_wait(bad), 0)
        self.assertEqual(m.parse_admission_wait("2.5"), 2.5)

    def test_nonstreaming_queue_added_before_observation(self):
        tree = ast.parse((root / "managers/tokenizer_manager.py").read_text())
        fn = next(
            n
            for c in tree.body
            if isinstance(c, ast.ClassDef)
            for n in c.body
            if isinstance(n, ast.FunctionDef) and n.name == "collect_metrics"
        )
        for a in fn.args.args:
            a.annotation = None
        ns = {"os": os, "DisaggregationMode": types.SimpleNamespace(PREFILL="p")}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "metrics", "exec"), ns)
        manager = types.SimpleNamespace(
            metrics_collector=MagicMock(),
            enable_priority_scheduling=False,
            disaggregation_mode="d",
            _request_has_grammar=lambda obj: False,
        )
        manager.metrics_collector.labels = {}
        state = types.SimpleNamespace(
            obj=types.SimpleNamespace(stream=False),
            ttft_observed=False,
            last_completion_tokens=0,
            admission_wait_seconds=3.0,
            time_stats=MagicMock(),
            finished=True,
        )
        state.time_stats.get_first_token_latency.return_value = 2.0
        state.time_stats.get_e2e_latency.return_value = 8.0
        recv = types.SimpleNamespace(
            completion_tokens=[2],
            finished_reasons=[{"type": "stop"}],
            prompt_tokens=[12],
            cached_tokens=[0],
        )
        with patch.dict(os.environ, {"SGLANG_TRUST_SMG_ADMISSION_TIMING": "1"}):
            ns["collect_metrics"](manager, state, recv, 0, {"decode_throughput": 4})
        manager.metrics_collector.histogram_admission_inclusive_ttft.labels.return_value.observe.assert_called_once_with(
            5.0
        )
        manager.metrics_collector.histogram_admission_inclusive_e2e.labels.return_value.observe.assert_called_once_with(
            11.0
        )
        manager.metrics_collector.observe_time_to_first_token.assert_called_once_with(
            {}, 2.0, stream=False
        )
        manager.metrics_collector.observe_request_tpot.assert_called_once_with(
            {}, 0.25, stream=False
        )


if __name__ == "__main__":
    unittest.main()
