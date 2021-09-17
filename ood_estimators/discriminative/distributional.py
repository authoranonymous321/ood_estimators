from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

from .discriminative_estimator import DiscriminativeEstimator
from ..utils.augmentation import CyclicTranslator
from ..utils.logging_utils import best_val_idx


class DistributionalCoherence(DiscriminativeEstimator):
    aug_dataset_cache: Dict[str, str] = {}

    label = "DC"
    primary_metric = "cos_diff"

    def __init__(self, cyclic_translation_device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self.augmenter = CyclicTranslator(device=cyclic_translation_device)

    def evaluate(self, model: PreTrainedModel, eval_dataloader: DataLoader, **kwargs) -> Dict[str, float]:
        logit_diffs = []
        l2_diffs = []
        # dataset loaded by by __init__ of superclass EstimatorAbs
        for inputs in tqdm(eval_dataloader, desc="Distributional coherence: translating..."):
            orig_logits = self._get_logits(model, inputs)
            tokenizer = kwargs["tokenizer"]
            texts = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
            # translation is expensive and deterministic, so we translate once and use the cache afterwards
            try:
                aug_texts = [self.aug_dataset_cache[text] for text in texts]
            except KeyError:
                aug_texts = self.augmenter.translate(texts)
                for text, aug_text in zip(texts, aug_texts):
                    self.aug_dataset_cache[text] = aug_text

            aug_inputs = tokenizer(aug_texts, truncation=True, return_tensors="pt", padding=True)
            aug_logits = self._get_logits(model, aug_inputs)

            diff = 1 - torch.nn.CosineSimilarity(dim=0)(orig_logits, aug_logits)
            l2_diff = torch.hstack([torch.dist(orig_logits[i], aug_logits[i], p=2)
                                    for i, _ in enumerate(orig_logits)])
            logit_diffs.append(diff)
            l2_diffs.append(l2_diff)

        logit_diffs = torch.hstack(logit_diffs)
        l2_diffs = torch.hstack(l2_diffs)

        log = {"cos_diff": logit_diffs.mean().item(),
               "cos_std": torch.std(logit_diffs).item(),
               "l2_diff": l2_diffs.mean().item(),
               "l2_std": torch.std(l2_diffs).item()}

        self.augmenter.to("cpu")
        return self._dict_log(log, eval_dataloader.dataset)

    @staticmethod
    def pick_best_performing_checkpoint(performance_logs: Dict[int, float], early_stopping_patience: int = 10):
        return best_val_idx(performance_logs, picking_strategy="min")
