from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import stats
from transformers import pipeline

from mario_gpt.dataset import MarioDataset
from mario_gpt.utils import view_level

STATISTICS = {
    "enemy": np.array([1.0, 4.0, 9.0]),
    "pipe": np.array([0.0, 3.0, 7.0]),
    "block": np.array([50.0, 75.0, 176.0]),

    "coin": np.array([1.0, 20.0, 50.0]),
    "powerup": np.array([0.0, 4.0, 9.0]),
    "goomba": np.array([0.0, 3.0, 7.0]),
    "koopa": np.array([0.0, 3.0, 7.0]),
}

FEATURE_EXTRACTION_MODEL = "facebook/bart-base"

# TODO: If this actually becomes a paper, have a "class_names" dict instad of making functions for each class
class Prompter:
    def __init__(
        self,
        level_tokenizer,
        prompter_model: str = FEATURE_EXTRACTION_MODEL,
        use_raw_counts: bool = False,
        statistics: Optional[Dict[str, Any]] = None,
    ):
        self.prompter_model = prompter_model
        self.feature_extraction = pipeline(
            "feature-extraction",
            model=prompter_model,
            tokenizer=prompter_model,
            framework="pt",
        )

        self.level_tokenizer = level_tokenizer

        self.use_raw_counts = use_raw_counts
        self.statistics = statistics
        if statistics is None:
            self.statistics = STATISTICS

        self.entity_chars = {
            "pipe": ["<>", "()"],
            "enemy": ["E", "B", "y"],
            "block": ["X", "S", "Q", "!", "2", "C", "#"],
            "koopa": ["r", "R", "k", "K"],
            "goomba": ["g", "G"],
            "powerup": ["1", "?", "U", "L"],
            "coin": ["o"],
        }

    def get_thresholds(self, entity_type: str) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics[entity_type]
        keywords = ["no", "little", "some", "many"]
        if entity_type == "block":
            keywords = ["little", "little", "some", "many"]
        return thresholds, keywords

    def count_entities(self, flattened_level: str, entity_type: str) -> int:
        if entity_type not in self.entity_chars:
            raise ValueError(f"Unknown entity type: {entity_type}")
            
        chars = self.entity_chars[entity_type]
        return sum(flattened_level.count(char) for char in chars)

    def generate_prompt(self, entity_type: str, flattened_level: str, level: str = None) -> Tuple[str, str]:
        if entity_type == "elevation":
            return self.elevation_prompt(flattened_level, level)
            
        count = self.count_entities(flattened_level, entity_type)
        keyword = f"{count}"
        
        if not self.use_raw_counts:
            thresholds, keywords = self.get_thresholds(entity_type)
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
            
        # Handle special plural cases
        plural = "enemies" if entity_type == "enemy" else f"{entity_type}s"
        return f"{keyword} {plural}", keyword

    def elevation_prompt(self, flattened_level: str, level: str):
        top_levels = level[:6]
        for t in top_levels:
            if "X" in t or "<" in t or ">" in t:
                return "high elevation", "high"
        return "low elevation", "low"

    def _flatten_level(self, string_level: List[str]) -> str:
        return "".join(string_level)

    def output_hidden(self, prompt: str, device: torch.device = torch.device("cpu")):
        return (
            self.feature_extraction(prompt, return_tensors="pt")[0]
            .mean(0)
            .to(device)
            .view(1, -1)
        )

    def dataset_statistics(self, dataset: MarioDataset):
        enemy_counts = []
        pipe_counts = []
        block_counts = []
        goomba_counts = []
        koopa_counts = []
        coin_counts = []
        powerup_counts = []
        for i in range(len(dataset)):
            level, _ = dataset[i]
            str_level = self._flatten_level(view_level(level, dataset.tokenizer))

            enemy_count = self.count_entities(str_level, "enemy")
            pipe_count = self.count_entities(str_level, "pipe")
            block_count = self.count_entities(str_level, "block")
            goomba_count = self.count_entities(str_level, "goomba")
            koopa_count = self.count_entities(str_level, "koopa")
            coin_count = self.count_entities(str_level, "coin")
            powerup_count = self.count_entities(str_level, "powerup")

            enemy_counts.append(enemy_count)
            pipe_counts.append(pipe_count)
            block_counts.append(block_count)
            goomba_counts.append(goomba_count)
            koopa_counts.append(koopa_count)
            coin_counts.append(coin_count)
            powerup_counts.append(powerup_count)
        d = {"enemy": {}, "pipe": {}, "block": {}, "goomba": {}, "koopa": {}, "coin": {}, "powerup": {}}

        d["enemy"] = stats.mstats.mquantiles(enemy_counts, [0.33, 0.66, 0.95])
        d["pipe"] = stats.mstats.mquantiles(pipe_counts, [0.33, 0.66, 0.95])
        d["block"] = stats.mstats.mquantiles(block_counts, [0.33, 0.66, 0.95])
        d["goomba"] = stats.mstats.mquantiles(goomba_counts, [0.33, 0.66, 0.95])
        d["koopa"] = stats.mstats.mquantiles(koopa_counts, [0.33, 0.66, 0.95])
        d["coin"] = stats.mstats.mquantiles(coin_counts, [0.33, 0.66, 0.95])
        d["powerup"] = stats.mstats.mquantiles(powerup_counts, [0.33, 0.66, 0.95])
        return d

    def __call__(
        self, level: torch.Tensor = None, sample_prompt: bool = False
    ) -> Union[str, torch.Tensor]:
        device: torch.device = torch.device("cpu")
        prompt_dict = {}
        
        if not sample_prompt:
            if level is None:
                raise ValueError("Level must be provided if sample_prompt is not true!")
            str_level = view_level(level, self.level_tokenizer)
            flattened_level = self._flatten_level(str_level)
            device = level.device

            # Generate prompts for all entity types
            for entity_type in self.entity_chars.keys():
                prompt, keyword = self.generate_prompt(entity_type, flattened_level, str_level)
                prompt_dict[entity_type] = prompt

            # Handle elevation separately
            elevation_prompt, elevation_keyword = self.elevation_prompt(flattened_level, str_level)
            prompt_dict["elevation_prompt"] = elevation_prompt
        else:
            str_level = None
            # Generate random prompts for all entity types
            for entity_type in self.entity_chars.keys():
                keywords = ["no", "little", "some", "many"]
                if entity_type == "block":
                    keywords = ["little", "little", "some", "many"]
                keyword = random.choice(keywords)
                prompt_dict[entity_type] = f"{keyword} {entity_type}s"

            # Handle elevation separately
            elevation_keyword = random.choice(["low", "high"])
            prompt_dict["elevation_prompt"] = f"{elevation_keyword} elevation"

        # Combine all prompts
        prompt = ", ".join(prompt_dict.values())
        # hidden = self.output_hidden(prompt, device=device)
        # return prompt, hidden, prompt_dict, str_level
        return prompt, prompt_dict, str_level

