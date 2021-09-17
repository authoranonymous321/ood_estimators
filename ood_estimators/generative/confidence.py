from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

from ..utils.logging_utils import best_val_idx
from ..datasets.custom_datasets import CustomDataset
from .generative_estimator import GenerativeEstimator


class GenerationConfidence(GenerativeEstimator):

    label = "GC"
    primary_metric = "std_dist"

    def __init__(self, ood_val_dataset: CustomDataset,
                 num_topn_outputs: int = 10, **kwargs):
        self.ood_dataset = ood_val_dataset
        self.num_topn_outputs = num_topn_outputs
        super().__init__(**kwargs)

    def _gather_scores_per_dataloader(self, model: PreTrainedModel, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = []
        for batch in tqdm(loader, desc="Translation for %s Generation Confidence" %
                                       loader.dataset.in_or_out_domain):
            outputs, scores_batch = self._get_all_outputs_scores(model, batch)
            scores.append(scores_batch)
        # omit outputs, where a number of retrieved scores is different, than requested
        expected_len = scores[0].shape[0]
        scores = [s_batch for s_batch in scores if s_batch.shape[0] == expected_len]

        ranges = torch.vstack(scores).max(dim=1).values - torch.vstack(scores).min(dim=1).values
        return torch.hstack(scores), ranges

    @staticmethod
    def _pairwise_matrix_cossims(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert len(a.shape) == len(b.shape) == 2
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]

        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def evaluate(self, model: PreTrainedModel, eval_dataloader: DataLoader, **kwargs) -> Dict[str, float]:
        ood_dataloader = DataLoader(
            self.ood_dataset,
            batch_size=eval_dataloader.batch_size,
            drop_last=eval_dataloader.drop_last,
            num_workers=eval_dataloader.num_workers,
            pin_memory=eval_dataloader.pin_memory,
        )
        id_scores, id_ranges = self._gather_scores_per_dataloader(model, eval_dataloader)
        ood_scores, ood_ranges = self._gather_scores_per_dataloader(model, ood_dataloader)

        id_log = {
            "mean": id_scores.mean().item(),
            "std": id_scores.std().item(),
            "range": id_ranges.mean().item(),

            "means_dist": id_scores.mean().item() - ood_scores.mean().item(),
            "std_dist": id_scores.std().item() - ood_scores.std().item(),
        }

        ood_log = {
            "mean": ood_scores.mean().item(),
            "std": ood_scores.std().item(),
            "range": ood_ranges.mean().item(),
        }

        return {**self._dict_log(id_log, eval_dataloader.dataset), **self._dict_log(ood_log, self.ood_dataset)}

    @staticmethod
    def pick_best_performing_checkpoint(performance_logs: Dict[int, float]):
        return best_val_idx(performance_logs, picking_strategy="min", warmup=5000)
