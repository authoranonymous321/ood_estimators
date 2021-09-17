import copy
from typing import Optional, List, Callable, Dict, Tuple, Union, Iterable

import numpy as np
import spacy as spacy
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM


class DropoutModelCustomizer:

    def __init__(self, model: PreTrainedModel, buffer_device: Optional[str] = None):
        self.orig_device = model.device
        buffer_device = model.device if buffer_device is None else buffer_device
        self.orig_model = model.to(buffer_device)
        self.custom_model = copy.deepcopy(self.orig_model)

    def set_random_dropout_model(self, dropout_ratio: float) -> PreTrainedModel:
        for layer in self.custom_model.base_model.modules():
            if type(layer) == torch.nn.Dropout:
                layer.dropout = dropout_ratio
                layer.train()
        return self.custom_model

    def reset(self) -> PreTrainedModel:
        del self.custom_model
        self.orig_model = self.orig_model.to(self.orig_device)
        return self.orig_model


class TransitionModel:

    def __init__(self, dataset: Union[Dataset, List[str]], lang: str):
        tagger = self._init_tagger(lang)

        self.corpus_words_tagged = [list(tagger(dataset[i])) for i in range(len(dataset))]
        corpus_tags = [[tag for token, tag in tagged_seq] for tagged_seq in self.corpus_words_tagged]
        self.all_tags, self.transition_probs = self._transition_graph_from_tags(corpus_tags)

    def distance(self, other: "TransitionModel") -> float:
        matching_indices = [i for i, tag in enumerate(self.all_tags) if tag in other.all_tags]
        self_transitions_subset = self.transition_probs[matching_indices, :][:, matching_indices]

        other_matching_indices = [i for i, tag in enumerate(other.all_tags) if tag in self.all_tags]
        other_transitions = other.transition_probs
        other_transitions_subset = other_transitions[other_matching_indices, :][:, other_matching_indices]

        return np.linalg.norm(self_transitions_subset - other_transitions_subset)

    @staticmethod
    def _transition_graph_from_tags(tag_sequences: List[List[str]]) -> Tuple[List[str], np.ndarray]:
        # construct 2-grams from sequences of tags and count an occurrence of each 2-gram for the transition graph
        counts: Dict[Tuple[str, str], int] = {}
        for sequence in tag_sequences:
            for i in range(2, len(sequence)):
                tags_from_to = tuple(sequence[i-2:i])
                try:
                    counts[tags_from_to] += 1
                except KeyError:
                    counts[tags_from_to] = 1
        all_tags = sorted(set([k[0] for k in counts.keys()] + [k[0] for k in counts.keys()]))
        transition_matrix = [[counts.get((tag_x, tag_y), 0) for tag_x in all_tags] for tag_y in all_tags]
        if not transition_matrix:
            # text is a single-word tag - can happen in initial training phases
            # we need to keep the dimensionality
            transition_matrix = [[]]

        transition_matrix_np = np.array(transition_matrix)
        return all_tags, transition_matrix_np / max(transition_matrix_np.sum(), 1)

    @staticmethod
    def _init_tagger(lang: str) -> Callable[[str], Iterable[Tuple[str, str]]]:
        if lang == "no":
            model_id = "nb_core_news_sm"
        elif lang == "en":
            model_id = "en_core_web_sm"
        elif lang == "de":
            model_id = "de_core_news_sm"
        else:
            raise ValueError("Language '%s' has no defined tagger" % lang)

        try:
            spacy_tagger = spacy.load(model_id)
        except OSError:
            # tagger not-yet downloaded
            # spacy.cli.download(model_id, False, "-q")
            spacy.cli.download(model_id)
            spacy_tagger = spacy.load(model_id)

        def _spacy_pos_tagger_wrapper(text: str) -> Iterable[Tuple[str, str]]:
            tokens_tagged = spacy_tagger(text)
            for token in tokens_tagged:
                yield token.text, token.pos_

        return _spacy_pos_tagger_wrapper


def get_initial_tokenizer_model(src_lang_iso: str,
                                tgt_lang_iso: str,
                                reinit_weights: bool) -> Tuple[PreTrainedTokenizer, AutoModelForSeq2SeqLM]:
    # Initialize to MarianMT architecture
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-%s-%s" % (src_lang_iso, tgt_lang_iso))
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-%s-%s" % (src_lang_iso, tgt_lang_iso))
    if reinit_weights:
        def reinit_model_weights(m: torch.nn.Module):
            if hasattr(m, "children"):
                for m_child in m.children():
                    if hasattr(m_child, "reset_parameters"):
                        # TODO: replace to total init, remove the latter:
                        m_child.reset_parameters()
                        # m_child.weight = torch.nn.Parameter(m_child.weight + 0.01)
                    reinit_model_weights(m_child)

        torch.manual_seed(2021)
        model.model.apply(reinit_model_weights)

    return tokenizer, model



