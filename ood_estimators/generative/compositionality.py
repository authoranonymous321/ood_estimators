from typing import Dict, Tuple, List, Callable

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..utils.logging_utils import best_val_idx
from ..datasets.opus import SRC_LANG, TGT_LANG
from ..utils.model_utils import TransitionModel
from ..datasets.custom_datasets import CustomDataset
from .generative_estimator import GenerativeEstimator


class SyntacticCompositionality(GenerativeEstimator):

    pos_tagger: Callable[[str], Tuple[str, str]]
    label = "SC"
    primary_metric = "translation_distance_diff_to_ood"

    def __init__(self, ood_val_dataset: CustomDataset, **kwargs):
        """
        Compares syntactic compositionality's perplexity on train distribution and outer distribution.
        Syntactic compositionality is a transition matrix of PoS tags
        :param ood_val_dataset:
        :param kwargs:
        """
        self.ood_dataset = ood_val_dataset

        super().__init__(**kwargs)

    def evaluate(self, model: PreTrainedModel, eval_dataloader: DataLoader, **kwargs) -> Dict[str, float]:
        id_texts = self._inputs_for_dataloader(eval_dataloader, kwargs["tokenizer"])
        id_base_transitions = TransitionModel(id_texts, SRC_LANG)

        id_translations = self._outputs_for_dataloader(eval_dataloader, model, kwargs["tokenizer"])
        id_translated_transitions = TransitionModel(id_translations, TGT_LANG)

        ood_dataloader = DataLoader(
            self.ood_dataset,
            batch_size=eval_dataloader.batch_size,
            drop_last=eval_dataloader.drop_last,
            num_workers=eval_dataloader.num_workers,
            pin_memory=eval_dataloader.pin_memory,
        )
        ood_texts = self._inputs_for_dataloader(ood_dataloader, kwargs["tokenizer"])
        ood_base_transitions = TransitionModel(ood_texts, SRC_LANG)

        ood_translations = self._outputs_for_dataloader(ood_dataloader, model, kwargs["tokenizer"])
        ood_translated_transitions = TransitionModel(ood_translations, TGT_LANG)

        id_transition_distance = id_base_transitions.distance(id_translated_transitions)
        ood_transition_distance = ood_base_transitions.distance(ood_translated_transitions)

        base_distance = id_base_transitions.distance(ood_base_transitions)
        translated_distance = id_translated_transitions.distance(ood_translated_transitions)

        id_log = {
            "translation_distance": id_transition_distance,
            "translation_distance_diff_to_ood": id_transition_distance - ood_transition_distance,
            "translation_distance_norm_diff_to_ood": id_transition_distance / max(ood_transition_distance, 0.1),
            "base_diff_to_ood": base_distance,
            "mutual_translations_distance_diff_to_ood": translated_distance
        }

        ood_log = {"translation_distance": ood_transition_distance}

        return {**self._dict_log(id_log, eval_dataloader.dataset), **self._dict_log(ood_log, self.ood_dataset)}

    def _inputs_for_dataloader(self, loader: DataLoader, tokenizer: PreTrainedTokenizer) -> List[str]:
        input_texts = []
        for batch in loader:
            input_texts.extend(self._decode_ids(batch["input_ids"], tokenizer))
        return input_texts

    def _outputs_for_dataloader(self, loader: DataLoader, model: PreTrainedModel,
                                tokenizer: PreTrainedTokenizer) -> List[str]:
        translated_texts = []
        for batch in tqdm(loader, desc="Translation for %s Generation Confidence" % loader.dataset.in_or_out_domain):
            outputs = self._get_all_outputs(model, batch)
            translated_texts.extend(self._decode_ids(outputs, tokenizer))
        return translated_texts

    @staticmethod
    def pick_best_performing_checkpoint(performance_logs: Dict[int, float]):
        return best_val_idx(performance_logs, picking_strategy="min", warmup=5000)
