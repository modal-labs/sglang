from sglang.srt.environ import envs
from sglang.srt.utils.common import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class DFlashServerBase(CustomTestCase):
    target_model = DEFAULT_TARGET_MODEL_DFLASH
    draft_model = DEFAULT_DRAFT_MODEL_DFLASH
    spec_algo = "DFLASH"
    spec_block_size = 16
    extra_args = []

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_SPEC_NAN_DETECTION.override(
            True
        ), envs.SGLANG_SPEC_OOB_DETECTION.override(True):
            cls.process = popen_launch_server(
                cls.target_model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    f"--speculative-algorithm={cls.spec_algo}",
                    f"--speculative-draft-model-path={cls.draft_model}",
                    f"--speculative-num-draft-tokens={cls.spec_block_size}",
                ]
                + cls.extra_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
