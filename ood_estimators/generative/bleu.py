from typing import Dict

import torch
from sacrebleu import corpus_bleu
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

from ..utils.logging_utils import best_val_idx
from ..datasets.custom_datasets import TranslationDataset
from .generative_estimator import GenerativeEstimator


class IDBLEU(GenerativeEstimator):

    label = "ID_BLEU"
    primary_metric = "val_BLEU"

    def evaluate(self, model: PreTrainedModel, eval_dataloader: DataLoader,
                 compute_loss: bool = False, **kwargs) -> Dict[str, float]:
        trues = []
        preds = []
        losses = []

        loss_func = CrossEntropyLoss()
        for batch in tqdm(eval_dataloader, desc="Inference for %s accuracy and loss" %
                                                eval_dataloader.dataset.in_or_out_domain):
            outputs = self._generate(model, batch)
            if compute_loss:
                new_losses = self._generation_loss(model, batch, loss_func)
                losses.append(new_losses)

            preds.extend(kwargs["tokenizer"].batch_decode(outputs.sequences, skip_special_tokens=True))
            trues.extend(kwargs["tokenizer"].batch_decode([[l_i for l_i in l if l_i != -100] for l in batch['labels']],
                                                          skip_special_tokens=True))

        bleu_score = corpus_bleu(preds, [trues]).score

        log = {"BLEU": bleu_score / 100}

        if compute_loss:
            mean_loss = torch.hstack(losses).mean().item()
            log["Loss"] = mean_loss

        return self._dict_log(log, eval_dataloader.dataset)

    @staticmethod
    def pick_best_performing_checkpoint(performance_logs: Dict[int, float]):
        return best_val_idx(performance_logs, picking_strategy="max", warmup=5000)


class OODBLEU(IDBLEU):

    label = "OOD_BLEU"

    def __init__(self, ood_val_dataset: TranslationDataset, **kwargs):
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


class TestBLEU(IDBLEU):

    def __init__(self, test_dataset: TranslationDataset, **kwargs):
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

