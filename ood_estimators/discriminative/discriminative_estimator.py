from abc import ABCMeta

import torch
from transformers import PreTrainedModel, BatchEncoding
from transformers.modeling_outputs import SequenceClassifierOutput

from ..estimator_abs import EstimatorAbs


class DiscriminativeEstimator(EstimatorAbs, metaclass=ABCMeta):

    @staticmethod
    def _get_logits(model: PreTrainedModel, inputs: BatchEncoding) -> torch.Tensor:
        return DiscriminativeEstimator._get_all_outputs(model, inputs).logits.detach()

    @staticmethod
    def _get_all_outputs(model: PreTrainedModel, inputs: BatchEncoding) -> SequenceClassifierOutput:
        return model(**{k: i.to(model.device) for k, i in inputs.items()})

