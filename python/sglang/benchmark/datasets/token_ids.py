import json
from argparse import Namespace
from dataclasses import dataclass
from typing import List

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow


@dataclass
class TokenIdsDataset(BaseDataset):
    dataset_path: str
    fixed_output_len: int
    num_prompts: int

    @classmethod
    def from_args(cls, args: Namespace) -> "TokenIdsDataset":
        return cls(
            dataset_path=args.dataset_path,
            fixed_output_len=args.sharegpt_output_len,
            num_prompts=args.num_prompts,
        )

    def load(self, tokenizer=None, model_id=None) -> List[DatasetRow]:
        rows = []
        with open(self.dataset_path) as f:
            for line in f:
                obj = json.loads(line)
                input_ids = obj["input_ids"]
                output_len = self.fixed_output_len or obj["output_len"]
                rows.append(
                    DatasetRow(
                        prompt=input_ids,
                        prompt_len=obj.get("prompt_len", len(input_ids)),
                        output_len=output_len,
                    )
                )
                if len(rows) == self.num_prompts:
                    break
        return rows
