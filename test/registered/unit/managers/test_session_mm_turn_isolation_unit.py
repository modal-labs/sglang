from array import array
from types import SimpleNamespace

import torch

from sglang.srt.managers.mm_utils import pad_input_ids_array
from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session, SessionController
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

VOCAB = 1 << 20
IMG_TOKENS = 766
IM_TOKEN_ID = 7


def _recv(
    rid,
    input_ids,
    *,
    parent_rid=None,
    mm_inputs=None,
    max_new_tokens=20,
):
    return SimpleNamespace(
        rid=rid,
        input_ids=array("q", input_ids),
        mm_inputs=mm_inputs,
        session_params=SimpleNamespace(
            id="s",
            rid=parent_rid,
            offset=None,
            replace=False,
            drop_previous_output=False,
        ),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        lora_id=None,
        custom_logit_processor=None,
        stream=False,
        return_logprob=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
        return_sampling_mask=False,
        require_reasoning=False,
        return_hidden_states=False,
        return_routed_experts=False,
        routed_experts_start_len=0,
        priority=None,
        evict_on_finish=False,
        routing_key=None,
        extra_key=None,
        http_worker_ipc=None,
        time_stats=None,
    )


def _image_turn(rid, *, prefix_len, pad_value, parent_rid=None):
    input_ids = [1] * prefix_len + [IM_TOKEN_ID] * IMG_TOKENS + [2] * 3
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=pad_value,
        pad_value=pad_value,
        offsets=[(prefix_len, prefix_len + IMG_TOKENS - 1)],
        feature=None,
    )
    image_inputs = MultimodalInputs(mm_items=[item], im_token_id=IM_TOKEN_ID)
    recv = _recv(
        rid,
        input_ids,
        parent_rid=parent_rid,
        mm_inputs=object(),
    )
    return recv, image_inputs


def _admit_image_turn(session, recv, image_inputs, *, vocab_size=VOCAB):
    req = session.create_req(recv, tokenizer=None, vocab_size=vocab_size)
    SessionController.adjust_mm_offsets(recv, req, image_inputs)
    req.set_origin_input_ids(pad_input_ids_array(req.origin_input_ids, image_inputs))
    req.extend_image_inputs(image_inputs)
    return req


def _placeholder_counts(req):
    input_ids = torch.tensor(list(req.origin_input_ids))
    return [
        int(torch.isin(input_ids, torch.tensor([item.pad_value])).sum())
        for item in req.multimodal_inputs.mm_items
    ]


class TestSessionMultimodalTurnIsolation(CustomTestCase):
    def test_streaming_abort_does_not_leak_items(self):
        session = Session(capacity_of_str_len=0, session_id="s", streaming=True)

        recv1, image1 = _image_turn("turn-1", prefix_len=5, pad_value=1_000_001)
        req1 = _admit_image_turn(session, recv1, image1)
        req1.output_ids.extend(range(20))
        session.finish_req(req1)
        committed_len = len(req1.origin_input_ids)

        recv2, image2 = _image_turn("turn-2", prefix_len=3, pad_value=1_000_002)
        req2 = _admit_image_turn(session, recv2, image2)
        session.abort_req(req2.rid)

        req3 = session.create_req(
            _recv("turn-3", [3] * 900, max_new_tokens=20),
            tokenizer=None,
            vocab_size=VOCAB,
        )

        self.assertEqual(len(req3.multimodal_inputs.mm_items), 1)
        self.assertEqual(_placeholder_counts(req3), [IMG_TOKENS])
        self.assertEqual(len(req1.multimodal_inputs.mm_items), 1)
        self.assertEqual(len(req3.origin_input_ids), committed_len + 20 + 900)

    def test_tree_siblings_do_not_share_item_lists(self):
        session = Session(capacity_of_str_len=0, session_id="s", streaming=False)

        recv1, image1 = _image_turn("turn-1", prefix_len=5, pad_value=1_000_001)
        req1 = _admit_image_turn(session, recv1, image1)
        req1.finished_reason = FINISH_LENGTH(1)
        req1.multimodal_inputs.mrope_position_delta = torch.zeros(1, dtype=torch.long)
        req1.multimodal_inputs.mrope_position_delta_repeated_cache = torch.zeros(
            3, 1, dtype=torch.long
        )

        recv2a, image2a = _image_turn(
            "turn-2a",
            prefix_len=3,
            pad_value=1_000_002,
            parent_rid=req1.rid,
        )
        req2a = _admit_image_turn(session, recv2a, image2a)
        self.assertEqual(len(req2a.multimodal_inputs.mm_items), 2)

        req2b = session.create_req(
            _recv("turn-2b", [3] * 4, parent_rid=req1.rid, mm_inputs=None),
            tokenizer=None,
            vocab_size=VOCAB,
        )

        self.assertEqual(len(req1.multimodal_inputs.mm_items), 1)
        self.assertEqual(len(req2b.multimodal_inputs.mm_items), 1)
        self.assertIsNone(req2a.multimodal_inputs.mrope_position_delta_repeated_cache)
        self.assertIsNone(req2b.multimodal_inputs.mrope_position_delta_repeated_cache)
        self.assertIsNotNone(req1.multimodal_inputs.mrope_position_delta_repeated_cache)
        self.assertEqual(_placeholder_counts(req2b), [IMG_TOKENS])

    def test_streaming_turn_pads_carried_fill_ids(self):
        session = Session(capacity_of_str_len=0, session_id="s", streaming=True)
        req1 = session.create_req(
            _recv("turn-1", [1] * 5, max_new_tokens=20),
            tokenizer=None,
            vocab_size=VOCAB,
        )
        req1.output_ids.extend(range(20))
        req1._refresh_fill_ids()
        self.assertEqual(len(req1.full_untruncated_fill_ids), 25)
        session.finish_req(req1)

        recv2, image2 = _image_turn("turn-2", prefix_len=3, pad_value=1_000_002)
        req2 = _admit_image_turn(session, recv2, image2)
        req2._refresh_fill_ids()

        self.assertEqual(
            list(req2.full_untruncated_fill_ids), list(req2.origin_input_ids)
        )
        self.assertEqual(
            int(
                torch.isin(
                    torch.tensor(list(req2.full_untruncated_fill_ids)),
                    torch.tensor([1_000_002]),
                ).sum()
            ),
            IMG_TOKENS,
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
