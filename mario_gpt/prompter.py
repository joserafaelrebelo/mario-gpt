from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.interpolate import splprep, splev
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
    

    def calculate_enemy_powerup_factor(self, level_data):
        # Define enemies and power-ups
        ENEMIES = {"g", "G", "k", "K", "r", "R", "y", "B", "b"}
        POWER_UPS = {"U", "?", "1"}

        enemy_count = sum(row.count(e) for e in ENEMIES for row in level_data)
        powerup_count = sum(row.count(p) for p in POWER_UPS for row in level_data)
        
        # Define weights for difficulty adjustment
        enemy_weight = 0.5  
        powerup_weight = 0.3  

        # print(f"Enemy Count: {enemy_count}, Power-Up Count: {powerup_count}")
        
        return enemy_weight * enemy_count - powerup_weight * powerup_count
    
    def process_level_difficulty(self, level_data):
        points = [(col_idx, row_idx) for row_idx, row in enumerate(level_data) for col_idx, char in enumerate(row) if char == 'P']
        
        if not points:
            return "Unknown"
        
        points = sorted(points, key=lambda p: p[0])
        points = np.array(points)
        x, y = points[:, 0], points[:, 1]
        
        try:
            tck, u = splprep([x, y], s=3)
        except Exception as e:
            print(f"Skipping segment due to insufficient points: {e}")
            return "Unknown"
        
        unew = np.linspace(0, 1, 1000)
        smooth_path = splev(unew, tck)
        xs, ys = smooth_path
        
        dx = np.gradient(xs, unew)
        dy = np.gradient(ys, unew)
        ddx = np.gradient(dx, unew)
        ddy = np.gradient(dy, unew)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        
        curvature_variation = np.trapz(np.abs(curvature), unew)
        vertical_range = np.max(ys) - np.min(ys)
        avg_slope = np.mean(np.abs(dy / (dx + 1e-6)))
        
        D = 0.2 * curvature_variation + 1.0 * vertical_range + 1.5 * avg_slope

        # Add enemy/power-up factor
        difficulty_factor = self.calculate_enemy_powerup_factor(level_data)
        D += difficulty_factor  # Increase or decrease D based on enemies and power-ups

        return "Easy" if D < 3.0 else "Medium" if D < 9 else "Hard"

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
                count = sum(flattened_level.count(char) for char in self.entity_chars[entity_type])
                prompt_dict[entity_type] = f"{count} {entity_type}s"
            
            difficulty = self.process_level_difficulty(str_level)
            prompt_dict["difficulty"] = difficulty

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

        prompt = ", ".join(prompt_dict.values())
        hidden = self.output_hidden(prompt, device=device)
        return prompt, hidden, prompt_dict, str_level

