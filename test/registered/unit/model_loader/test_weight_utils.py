"""Unit tests for srt/model_loader/weight_utils.py shard-index consistency."""

import json
import os
import tempfile
import unittest
from unittest import mock

from sglang.srt.model_loader import weight_utils
from sglang.srt.model_loader.weight_utils import filter_duplicate_safetensors_files
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

INDEX_NAME = "model.safetensors.index.json"


def _write_index(folder, weight_map):
    with open(os.path.join(folder, INDEX_NAME), "w") as f:
        json.dump({"weight_map": weight_map}, f)


def _touch(folder, name):
    path = os.path.join(folder, name)
    open(path, "w").close()
    return path


class TestFilterDuplicateSafetensorsFiles(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_shard_raises(self):
        # Index lists two shards, only one on disk (interrupted download).
        _write_index(
            self.folder,
            {
                "w1": "model-00001-of-00002.safetensors",
                "w2": "model-00002-of-00002.safetensors",
            },
        )
        present = _touch(self.folder, "model-00001-of-00002.safetensors")

        with self.assertRaises(RuntimeError) as cm:
            filter_duplicate_safetensors_files(
                hf_weights_files=[present],
                hf_folder=self.folder,
                index_file=INDEX_NAME,
            )
        self.assertIn("model-00002-of-00002.safetensors", str(cm.exception))

    def test_complete_checkpoint_filters_non_indexed(self):
        # All indexed shards present; a non-indexed duplicate is still filtered out.
        _write_index(
            self.folder,
            {
                "w1": "model-00001-of-00002.safetensors",
                "w2": "model-00002-of-00002.safetensors",
            },
        )
        shard1 = _touch(self.folder, "model-00001-of-00002.safetensors")
        shard2 = _touch(self.folder, "model-00002-of-00002.safetensors")
        extra = _touch(self.folder, "consolidated.safetensors")

        result = filter_duplicate_safetensors_files(
            hf_weights_files=[shard1, shard2, extra],
            hf_folder=self.folder,
            index_file=INDEX_NAME,
        )
        self.assertEqual(sorted(result), sorted([shard1, shard2]))

    def test_single_file_model_no_index_returns_unchanged(self):
        # No index on disk (single-file / dummy / object-storage): early return.
        single = _touch(self.folder, "model.safetensors")

        result = filter_duplicate_safetensors_files(
            hf_weights_files=[single],
            hf_folder=self.folder,
            index_file=INDEX_NAME,
        )
        self.assertEqual(result, [single])


class TestFastsafetensorsNoGDS(CustomTestCase):
    def test_nogds_is_enabled_only_by_exact_env_value(self):
        loader_calls = []

        class FakeProcessGroup:
            def rank(self):
                return 0

            def size(self):
                return 1

        class FakeFileBuffer:
            def __init__(self):
                self.key_to_rank_lidx = {}

        class FakeLoader:
            def __init__(self, process_group, device, *, nogds):
                loader_calls.append(nogds)

            def add_filenames(self, rank_file_map):
                pass

            def copy_files_to_device(self):
                return FakeFileBuffer()

            def close(self):
                pass

        def run_with_env(value):
            if value is None:
                os.environ.pop("SGLANG_FASTSAFETENSORS_NOGDS", None)
            else:
                os.environ["SGLANG_FASTSAFETENSORS_NOGDS"] = value

            with (
                mock.patch.object(weight_utils, "SafeTensorsFileLoader", FakeLoader),
                mock.patch.object(
                    weight_utils, "SingleGroup", return_value=FakeProcessGroup()
                ),
                mock.patch.object(
                    weight_utils.torch.distributed,
                    "is_initialized",
                    return_value=False,
                ),
                mock.patch.object(
                    weight_utils,
                    "tqdm",
                    side_effect=lambda iterable, **_: iterable,
                ),
            ):
                list(
                    weight_utils.fastsafetensors_weights_iterator(["model.safetensors"])
                )

        with mock.patch.dict(os.environ, {}, clear=False):
            for value, expected in (
                (None, False),
                ("0", False),
                ("true", False),
                ("1", True),
            ):
                with self.subTest(value=value):
                    run_with_env(value)
                    self.assertEqual(loader_calls[-1], expected)


if __name__ == "__main__":
    unittest.main()
