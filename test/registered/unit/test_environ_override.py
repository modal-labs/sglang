"""EnvField.override restores the prior state on every exit path, so a test
body that raises cannot leak its override into later tests."""

import os
import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class EnvFieldOverrideTest(CustomTestCase):
    def test_override_restores_unset_state_when_body_raises(self):
        field = envs.SGLANG_ENABLE_KV_RESERVED_SLOT_WRITE_GUARD
        field.clear()
        self.assertFalse(field.is_set())

        with self.assertRaises(RuntimeError):
            with field.override(False):
                self.assertFalse(field.get())
                raise RuntimeError("body failed")

        self.assertFalse(field.is_set())
        self.assertTrue(field.get())

    def test_override_restores_previous_value_when_body_raises(self):
        field = envs.SGLANG_ENABLE_KV_ZERO_PAGES_ON_ALLOC
        with field.override(False):
            with self.assertRaises(RuntimeError):
                with field.override(True):
                    self.assertTrue(field.get())
                    raise RuntimeError("body failed")
            self.assertEqual(os.environ[field.name], "False")
            self.assertFalse(field.get())

    def test_override_restores_explicit_none_when_body_raises(self):
        field = envs.SGLANG_TEST_MAX_RETRY
        with field.override(None):
            self.assertTrue(field.is_set())
            self.assertIsNone(field.get())
            with self.assertRaises(RuntimeError):
                with field.override(3):
                    self.assertEqual(field.get(), 3)
                    raise RuntimeError("body failed")
            self.assertTrue(field.is_set())
            self.assertIsNone(field.get())


if __name__ == "__main__":
    unittest.main()
