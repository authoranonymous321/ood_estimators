from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

from ..utils.logging_utils import best_val_idx
from .discriminative_estimator import DiscriminativeEstimator
from ..utils.model_utils import DropoutModelCustomizer


class PredictionConsistency(DiscriminativeEstimator):

    label = "PC"
    primary_metric = "ratio"

    def evaluate(self, model: PreTrainedModel, eval_dataloader: DataLoader,
                 pc_r: int = 10, pc_dropout: float = 0.4, pc_buffer_device: str = None, **kwargs) -> Dict[str, float]:
        model_customizer = DropoutModelCustomizer(model, pc_buffer_device)
        r_predictions = []
        for r_i in range(pc_r):
            inference_model = model_customizer.set_random_dropout_model(dropout_ratio=pc_dropout)
            logits = []
            for batch in tqdm(eval_dataloader, desc="Inference with 'dropout' model %s/%s" % (r_i, pc_r)):
                logits.append(self._get_logits(inference_model, batch))
            predictions = torch.vstack(logits).argmax(dim=-1)

            r_predictions.append(predictions)

        all_predictions = torch.vstack(r_predictions).cpu()
        consistent_predictions_count = torch.sum(all_predictions == all_predictions.mode(dim=0).values, dim=0)
        rel_consistence = consistent_predictions_count / pc_r
        mean_consistence = rel_consistence.mean()
        consistence_std = torch.std(rel_consistence)

        log = {"ratio": mean_consistence.item(),
               "std": consistence_std.item()}

        model_customizer.reset()

        return self._dict_log(log, eval_dataloader.dataset)

    @staticmethod
    def pick_best_performing_checkpoint(performance_logs: Dict[int, float]):
        # system is initialized confident, then adjusts logits to its perplexity and then returns to a confident state
        confidence_minima_idx = sorted(performance_logs.keys(), key=lambda k: performance_logs[k])[0]
        confident_interval = {k: val for k, val in performance_logs.items() if k > confidence_minima_idx}

        return best_val_idx(confident_interval, picking_strategy="max")
