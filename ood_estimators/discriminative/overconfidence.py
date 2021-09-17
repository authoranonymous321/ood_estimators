from typing import Dict

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from ..utils.logging_utils import best_val_idx
from .discriminative_estimator import DiscriminativeEstimator


class Overconfidence(DiscriminativeEstimator):

    label = "OC"
    primary_metric = "ratio"

    def __init__(self, threshold: int = 0.8, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def evaluate(self, model: PreTrainedModel, eval_dataloader: DataLoader, **kwargs) -> Dict[str, float]:
        logits_all = []
        over_threshold = 0
        over_threshold_agg = 0
        for inputs in eval_dataloader:
            logits = self._get_logits(model, inputs)
            over_threshold += torch.sum(logits.max() > self.threshold).item()
            over_threshold_agg += torch.sum(logits.max()[logits.max() > self.threshold] - self.threshold).item()
            logits_all.append(logits)
        logits_all_t = torch.vstack(logits_all)

        log = {"ratio": (over_threshold / len(logits_all_t)),
               "sum": (over_threshold_agg / len(logits_all_t)),
               "max_logits_mean": logits_all_t.max(dim=1).values.mean().item(),
               "max_logits_std": torch.std(logits_all_t.max(dim=1).values).item()}

        return self._dict_log(log, eval_dataloader.dataset)

    @staticmethod
    def pick_best_performing_checkpoint(performance_logs: Dict[int, float]) -> int:
        # system is initialized confident, then adjusts logits to its perplexity and then returns to a confident state
        confidence_minima_idx = sorted(performance_logs.keys(), key=lambda k: performance_logs[k])[0]
        confident_interval = {k: val for k, val in performance_logs.items() if k > confidence_minima_idx}

        return best_val_idx(confident_interval, picking_strategy="max")
