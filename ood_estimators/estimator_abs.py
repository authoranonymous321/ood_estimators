from abc import ABCMeta, abstractmethod
from typing import Dict, Callable

from torch.utils.data import DataLoader
from transformers import TrainerCallback, PreTrainedModel, TrainingArguments, TrainerState, TrainerControl, \
    PreTrainedTokenizer

from .datasets.custom_datasets import CustomDataset
from .utils.logging_utils import log_callback


class OODCallback(TrainerCallback):

    def __init__(self, eval_method: Callable[[PreTrainedModel, PreTrainedTokenizer], Dict[str, float]]):
        super().__init__()
        self.eval_method = eval_method
        pass

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        metrics = self.eval_method(**kwargs)
        log_callback(metrics, state, args.logging_dir)


class EstimatorAbs(metaclass=ABCMeta):

    label: str
    primary_metric: str

    def __init__(self, **kwargs):
        pass

    def _dict_log(self, logs: Dict[str, float], dataset: CustomDataset) -> Dict[str, float]:
        prefix = "%s_%s_%s_" % (dataset.in_or_out_domain, self.label, dataset.split)
        return {prefix + k: v for k, v in logs.items()}

    def get_callback(self) -> OODCallback:
        class Callback(OODCallback):
            pass
        callback = Callback(self.evaluate)

        # HF Trainer does not accept multiple callbacks of the same type, so we need to differentiate their names
        Callback.__name__ += self.__class__.__name__

        return callback

    @abstractmethod
    def evaluate(self, model: PreTrainedModel, eval_loader: DataLoader, **kwargs) -> Dict[str, float]:
        pass

    @staticmethod
    @abstractmethod
    def pick_best_performing_checkpoint(performance_logs: Dict[int, float]):
        pass
