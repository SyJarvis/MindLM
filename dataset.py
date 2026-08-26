import ast
import json
from pathlib import Path

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        super().__init__()
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        if tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer must define a pad token")
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        sample = self.df.iloc[index]
        text = str(sample['text'])
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0 means the padded token is excluded from the token-level loss.
        loss_mask = [1] * text_len + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)


class PackedPretrainDataset(Dataset):
    """Memory-map fixed-length, EOS-delimited pretraining token blocks.

    ``prepare_pretrain_data.py`` writes ``<prefix>.bin`` and ``<prefix>.json``.
    Every stored record has ``max_length + 1`` real tokens, so this dataset avoids
    padding and its loss mask is always one. It is intended for pretraining only;
    answer-only SFT masking remains in ``SFTDataset``.
    """

    FORMAT = "mindlm_packed_pretrain_v1"

    def __init__(self, prefix, max_length, tokenizer_vocab_size=None):
        super().__init__()
        prefix = Path(prefix)
        self.tokens_path = prefix.with_suffix(".bin")
        metadata_path = prefix.with_suffix(".json")
        if not self.tokens_path.is_file():
            raise FileNotFoundError(f"Packed token file not found: {self.tokens_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Packed metadata file not found: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("format") != self.FORMAT:
            raise ValueError(f"Unsupported packed dataset format in {metadata_path}")
        if metadata.get("sequence_length") != max_length:
            raise ValueError(
                f"Packed sequence length is {metadata.get('sequence_length')}, "
                f"but the model requires {max_length}."
            )
        if tokenizer_vocab_size is not None and metadata.get("tokenizer_vocab_size") != tokenizer_vocab_size:
            raise ValueError(
                "Packed dataset tokenizer vocabulary does not match the selected tokenizer: "
                f"{metadata.get('tokenizer_vocab_size')} != {tokenizer_vocab_size}."
            )

        self.max_length = max_length
        self.tokens_per_record = metadata.get("tokens_per_record")
        self.num_sequences = metadata.get("num_sequences")
        dtype = np.dtype(metadata.get("dtype", "uint32"))
        if self.tokens_per_record != max_length + 1:
            raise ValueError("Packed dataset record width must be max_length + 1")
        if not isinstance(self.num_sequences, int) or self.num_sequences < 1:
            raise ValueError("Packed dataset contains no full token sequences")
        expected_bytes = self.num_sequences * self.tokens_per_record * dtype.itemsize
        if self.tokens_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"Packed token file size does not match metadata: expected {expected_bytes}, "
                f"got {self.tokens_path.stat().st_size}."
            )

        self.metadata = metadata
        self.tokens = np.memmap(
            self.tokens_path,
            mode="r",
            dtype=dtype,
            shape=(self.num_sequences, self.tokens_per_record),
        )
        self.loss_mask = torch.ones(max_length, dtype=torch.int64)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index: int):
        record = self.tokens[index]
        # Copy before converting because a read-only memmap cannot safely back a
        # mutable torch Tensor in a DataLoader worker.
        input_ids = torch.from_numpy(np.asarray(record[:-1], dtype=np.int64).copy())
        targets = torch.from_numpy(np.asarray(record[1:], dtype=np.int64).copy())
        return input_ids, targets, self.loss_mask


class SFTDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=1024, prompt_max_len=512, answer_max_len=256):
        super().__init__()
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        if tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer must define a pad token")
        self.df = df
        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        self.tokenizer = tokenizer
        self.padding = tokenizer.pad_token_id
        marker = "<|im_start|>assistant\n" if "<|im_start|>" in getattr(
            tokenizer, "all_special_tokens", []
        ) else "<s>assistant\n"
        self.bos_id = self.tokenizer(marker, add_special_tokens=False).data['input_ids']
        if len(self.bos_id) >= max_length:
            raise ValueError("max_length must leave room for at least one assistant token")

    def __len__(self):
        return self.df.shape[0]

    def find_sublist_index(self, main_list, sub_list) -> int:
        last_index = -1
        for i in range(len(main_list) - len(sub_list) + 1):
            if main_list[i:i + len(sub_list)] == sub_list:
                last_index = i
        return last_index

    def safe_eval(self, s):
        if isinstance(s, list):
            return s
        if not isinstance(s, str):
            return []
        try:
            res = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return []
        return res if isinstance(res, list) else []

    def __getitem__(self, index: int):
        sample = self.df.iloc[index]
        history = self.safe_eval(sample['history'])
        q = str(sample['q'])
        a = str(sample['a'])

        messages = []
        for history_message in history:
            if len(history_message) <= 1:
                continue
            messages.append(
                {"role": 'user', "content": str(history_message[0])[:self.max_length // 2]}
            )
            messages.append(
                {"role": 'assistant', "content": str(history_message[1])[:self.max_length // 2]}
            )

        messages += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        new_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            # The final assistant message is part of the supervised sample;
            # appending a second generation marker would hide its boundary.
            add_generation_prompt=False,
        )
        full_input_ids = self.tokenizer(new_prompt).data['input_ids']
        marker_index = self.find_sublist_index(full_input_ids, self.bos_id)
        if marker_index < 0:
            raise ValueError(f"Sample {index} has no final assistant marker in its chat template")

        answer_start = marker_index + len(self.bos_id)
        answer_ids = full_input_ids[answer_start:]
        max_answer_length = self.max_length - len(self.bos_id)
        if len(answer_ids) > max_answer_length:
            input_id = self.bos_id + answer_ids[:max_answer_length]
        else:
            context_length = self.max_length - len(self.bos_id) - len(answer_ids)
            context_ids = full_input_ids[:marker_index][-context_length:] if context_length else []
            input_id = context_ids + self.bos_id + answer_ids

        question_length = len(input_id) - len(answer_ids[:max_answer_length])
        if question_length >= len(input_id):
            raise ValueError(f"Sample {index} has no assistant tokens after truncation")

        padding_len = self.max_length - len(input_id)
        input_id = input_id + [self.padding] * padding_len
        mask_len = self.max_length - question_length - padding_len
        loss_mask = [0] * question_length + [1] * (mask_len) + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        loss_mask_tensor = torch.from_numpy(loss_mask)

        return X_tensor, Y_tensor, loss_mask_tensor
