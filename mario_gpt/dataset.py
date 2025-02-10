from __future__ import annotations

from typing import List, Optional
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

# from mario_gpt.level import FULL_LEVEL_STR_WITH_PATHS
from mario_gpt.prompter import Prompter

DEFAULT_MODEL = "distilgpt2"

def split_given_size(a, size):
    return np.split(a, np.arange(size, len(a), size))

def flip_and_transpose(arr: np.array, flip_first: bool = False):
    if arr.shape[-1] > 1:
        if flip_first:
            return np.flip(arr, -1).transpose()
        return np.flip(arr.transpose(), -1)
    return arr

def join_list_of_list(str_lists):
    return ["".join(s) for s in str_lists]

def characterize(str_lists):
    return [list(s) for s in str_lists]

class MarioDataset(Dataset):
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        folder_path: Optional[str] = None,
        context_len: int = 700,
        height: int = 14,
        remove_start_end_tokens: bool = False,
        sample_all_indices: bool = False,
    ):
        if folder_path is None or not os.path.isdir(folder_path):
            raise ValueError("Invalid folder path provided.")

        self.folder_path = folder_path
        self.context_len = context_len
        self.height = height
        self.sample_all_indices = sample_all_indices

        level_tokenizer = AutoTokenizer.from_pretrained("shyamsn97/Mario-GPT2-700-context-length")
        self.prompter_base = Prompter(level_tokenizer=level_tokenizer)

        def get_training_corpus():
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".txt"):
                        with open(os.path.join(root, file), "r") as f:
                            lines = f.readlines()
                            if lines:  
                                level_text = "".join(lines[1:])

                                def transform_to_list(input_string):
                                    lines = input_string.strip().split('\n')
                                    return lines

                                level_list = transform_to_list(level_text)

                                tokenized_level = self.prompter_base.level_tokenizer(level_list, return_tensors="pt")
                                level_tensor = tokenized_level['input_ids']
                                flattened_tensor = level_tensor.view(-1)
                                prompt_base, _, _ = self.prompter_base(level=flattened_tensor)

                                combined_text = f"{level_text} <sep> {prompt_base}"
                                yield list(combined_text)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)

        if getattr(tokenizer, "train_new_from_iterator", None) is not None:
            print("Training tokenizer from iterator")
            tokenizer.add_special_tokens({"sep_token": "<sep>"})
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.tokenizer = tokenizer.train_new_from_iterator(
                get_training_corpus(), 52000
            )
        elif getattr(tokenizer, "train_from_iterator", None) is not None:
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
            tokenizer.add_special_tokens({"sep_token": "<sep>"})
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

            self.tokenizer = self.tokenizer.train_new_from_iterator(
                get_training_corpus(), 52000
            )
        else:
            self.tokenizer = tokenizer

        # Resize token embeddings to include <sep> token
        self.tokenizer.model_max_length += 1

        self.data = []
        self.character_set = set()
        current_id = 0

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        lines = f.readlines()[1:]
                        level_string = "".join(lines)

                    self.character_set.update(set(level_string) - {"\n"})
                    x, str_arr = self.convert_level_to_tensor(level_string.split("\n"))

                    input_ids = x["input_ids"].squeeze()
                    attention_masks = x["attention_mask"].squeeze()
                    if remove_start_end_tokens:
                        input_ids = input_ids[1:-1]
                        attention_masks = attention_masks[1:-1]

                    indices = self.generate_indices(input_ids, current_id)

                    self.data.append({
                        "input_ids": input_ids,
                        "attention_masks": attention_masks,
                        "indices": indices
                    })

                    current_id += len(input_ids) - context_len

        self.vocab_size = len(self.character_set)

        all_input_ids = torch.cat([data["input_ids"] for data in self.data])
        self.unique_tokens, self.unique_counts = all_input_ids.unique(return_counts=True)
        self.input_ids = all_input_ids
        self.weighted_unique_counts = (
            1.0 / self.unique_counts / torch.sum(self.unique_counts)
        )

        self.token_dict = {}
        string_tokens = list(self.tokenizer.decode(self.unique_tokens))
        for int_token, string_token in zip(self.unique_tokens, string_tokens):
            self.token_dict[string_token] = int_token

    def convert_level_to_tensor(self, level: List[str]):
        str_arr = flip_and_transpose(np.array(characterize(level)))
        str_arr = "".join(join_list_of_list(str_arr))

        x = self.tokenizer(str_arr, return_tensors="pt")
        return x, str_arr

    def generate_indices(self, input_ids, start_id):
        out = []
        for idx in range(input_ids.shape[0] - self.context_len):
            if idx % self.height == 0 or self.sample_all_indices:
                arange = torch.arange(idx, idx + self.context_len)  # Use local indices
                # if arange[-1] >= len(input_ids):
                    # print(f"Invalid range: {arange}, input_ids size: {len(input_ids)}")
                out.append(arange)
        return torch.stack(out)

    def sample_indices(self, batch_size):
        out = []
        for _ in range(batch_size):
            start_idx = np.random.randint(0, self.__len__() - self.context_len)
            indices = torch.arange(start_idx, start_idx + self.context_len)
            out.append(indices)
        return torch.stack(out)

    def __len__(self):
        return sum(len(data["indices"]) for data in self.data)

    def __getitem__(self, idx):
        for i, data in enumerate(self.data):
            num_indices = len(data["indices"])
            if idx < num_indices:
                indices = data["indices"][idx]
                # Log the indices for debugging
                # print(f"Accessing data block {i}, index {idx}, indices {indices}")
                return data["input_ids"][indices], data["attention_masks"][indices]
            idx -= num_indices
        raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}")


    def __str__(self):
        output = []
        for data in self.data:
            str_list = characterize(self.tokenizer.batch_decode(data["input_ids"]))
            string = "\n".join(
                join_list_of_list(flip_and_transpose(np.array(str_list), True))
            )
            output.append(string)
        return "\n---\n".join(output)
