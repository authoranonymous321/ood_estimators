import random
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import List, Iterable, Optional

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding


class DataSplit(Enum):
    train = 0
    val = 1
    test = 2
    end = 3


class IDSplitOffset(Enum):
    train = 0.0
    val = 0.8
    test = 0.9
    end = 1.0


class OODSplitOffset(Enum):
    train = -1
    val = 0.0
    test = 0.5
    end = 1.0


class CustomDataset(Dataset, metaclass=ABCMeta):
    dataset_id: str = None
    dataset_inputs: BatchEncoding = None

    @abstractmethod
    def __init__(self, in_or_out_domain: str, split: str, data_dir: str, tokenizer: PreTrainedTokenizer):
        assert in_or_out_domain in ["id", "ood"]
        self.in_or_out_domain = in_or_out_domain
        self.split = split
        self.data_dir = data_dir
        self.tokenizer = tokenizer

    def _get_in_split(self, data: List) -> List:
        domain_offset = IDSplitOffset if self.in_or_out_domain == "id" else OODSplitOffset
        offset_start_pos = int(len(data) * domain_offset[self.split].value)
        offset_end_pos = int(len(data) * list(domain_offset)[DataSplit[self.split].value + 1].value)
        return data[offset_start_pos: offset_end_pos]

    def _encode(self, texts: List[str]) -> BatchEncoding:
        return self.tokenizer(texts, truncation=True, return_tensors="pt", padding="max_length")

    @staticmethod
    def _disrupt_text(text: str, per_word_prob: float = 0.5) -> str:
        words = text.split()
        out_words = []
        for word in words:
            random_float = random.randint(0, 1000) / 1000
            if random_float < per_word_prob:
                if random_float < per_word_prob / 3:
                    # typos
                    char_to_drop = random.choice(word)
                    char_to_insert = random.choice(word)
                    out_words.append(word.replace(char_to_drop, char_to_insert, 1))
                elif random_float < per_word_prob * 2 / 3:
                    # missing characters
                    char_to_drop = random.choice(word)
                    out_words.append(word.replace(char_to_drop, "", 1))
                else:
                    # skip word
                    continue
            else:
                out_words.append(word)

        return " ".join(out_words)


class ClassificationDataset(CustomDataset, metaclass=ABCMeta):
    dataset_inputs = None
    dataset_targets = None

    @abstractmethod
    def get_targets(self) -> List[str]:
        pass

    def _init_target_mapping(self, targets: Iterable[str]):
        self.id_to_target = dict(enumerate(sorted(set(targets))))
        self.target_to_id = {v: k for k, v in self.id_to_target.items()}


class TranslationDataset(CustomDataset, metaclass=ABCMeta):

    def _encode_pairs(self, src_texts: List[str], tgt_texts: List[str], tokenizer: PreTrainedTokenizer,
                      exclude_padding_from_loss: Optional[bool] = True) -> BatchEncoding:

        model_inputs = self._encode(src_texts)
        label_ids = self._encode(tgt_texts).input_ids
        model_inputs['labels'] = label_ids

        # pad tokens are excluded from the loss
        if exclude_padding_from_loss:
            model_inputs['labels'][model_inputs['labels'] == tokenizer.pad_token_id] = -100

        return model_inputs
