from abc import ABCMeta
from typing import Tuple, Iterable, List, Union, Dict

import torch
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, BatchEncoding, PreTrainedTokenizer
from transformers.generation_utils import BeamSearchEncoderDecoderOutput

from ..estimator_abs import EstimatorAbs


class GenerativeEstimator(EstimatorAbs, metaclass=ABCMeta):

    @staticmethod
    def _generate(model: PreTrainedModel, inputs: Union[BatchEncoding, Dict[str, torch.Tensor]],
                  **generate_args) -> BeamSearchEncoderDecoderOutput:
        return model.generate(**{l: vals.to(model.device) for l, vals in inputs.items() if l != "labels"},
                              **generate_args, return_dict_in_generate=True)

    @staticmethod
    def _generation_loss(model: PreTrainedModel, inputs: Union[BatchEncoding, Dict[str, torch.Tensor]],
                         loss_func: CrossEntropyLoss) -> torch.Tensor:
        with torch.no_grad():
            lm_logits = model(**{l: vals.to(model.device) for l, vals in inputs.items() if l != "labels"}) + model.final_logits_bias
            masked_lm_loss = loss_func(lm_logits.view(-1, model.config.vocab_size), inputs["labels"].view(-1))
        return masked_lm_loss

    @staticmethod
    def _get_all_outputs(model: PreTrainedModel,
                         inputs: Union[BatchEncoding, Dict[str, torch.Tensor]]) -> torch.LongTensor:
        return GenerativeEstimator._generate(model, inputs).sequences.detach().cpu()

    @staticmethod
    def _get_all_outputs_scores(model: PreTrainedModel,
                                inputs: Union[BatchEncoding, Dict[str, torch.Tensor]],
                                num_topn_outputs: int = 5) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        beam_search_out = GenerativeEstimator._generate(model, inputs,
                                                        output_scores=True,
                                                        num_return_sequences=num_topn_outputs,
                                                        num_beams=num_topn_outputs)
        return beam_search_out.sequences.detach().cpu(), beam_search_out.sequences_scores.detach().cpu()

    @staticmethod
    def _decode_ids(ids_batch: Iterable[Iterable[int]], tokenizer: PreTrainedTokenizer) -> List[str]:
        raw_texts = tokenizer.batch_decode(ids_batch, skip_special_tokens=True)
        out_texts = [text.replace("▁", " ") for text in raw_texts]
        return out_texts
