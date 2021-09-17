from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

from ..datasets.custom_datasets import ClassificationDataset
from .discriminative_estimator import DiscriminativeEstimator
from ..utils.logging_utils import best_val_idx


class IDAccuracy(DiscriminativeEstimator):

    label = "Accuracy"
    primary_metric = "val_Accuracy"

    def evaluate(self, model: PreTrainedModel, eval_dataloader: DataLoader, **kwargs) -> Dict[str, float]:
        trues = []
        logits = []
        losses = []

        for batch in tqdm(eval_dataloader, desc="Inference for %s accuracy and loss" %
                                                eval_dataloader.dataset.in_or_out_domain):
            outputs = self._get_all_outputs(model, batch)
            logits.append(outputs.logits.detach().cpu())
            losses.append(outputs.loss.detach().cpu())
            trues.append(batch['labels'])
        predictions = torch.vstack(logits).argmax(dim=-1)
        trues_t = torch.hstack(trues)
        accuracy = torch.sum(predictions == trues_t) / len(predictions)

        mean_loss = torch.hstack(losses).mean().item()

        log = {"Accuracy": accuracy.item(),
               "Loss": mean_loss}

        return self._dict_log(log, eval_dataloader.dataset)

    @staticmethod
    def pick_best_performing_checkpoint(performance_logs: Dict[int, float]):
        return best_val_idx(performance_logs, picking_strategy="max")


class OODAccuracy(IDAccuracy):

    def __init__(self, ood_val_dataset: ClassificationDataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = ood_val_dataset

    def evaluate(self, model: PreTrainedModel, eval_dataloader: DataLoader, **kwargs) -> Dict[str, float]:
        custom_dataloader = DataLoader(
            self.dataset,
            batch_size=eval_dataloader.batch_size,
            drop_last=eval_dataloader.drop_last,
            num_workers=eval_dataloader.num_workers,
            pin_memory=eval_dataloader.pin_memory,
        )

        return super().evaluate(model, custom_dataloader, **kwargs)


class TestAccuracy(IDAccuracy):

    def __init__(self, test_dataset: ClassificationDataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = test_dataset

    def evaluate(self, model: PreTrainedModel, batch_size: int, **kwargs) -> Dict[str, float]:
        custom_dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            drop_last=True,
            pin_memory=True
        )

        return super().evaluate(model, custom_dataloader, **kwargs)

