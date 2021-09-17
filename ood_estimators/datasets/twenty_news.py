from typing import List, Iterable, Tuple, Dict

import torch
from sklearn.datasets import fetch_20newsgroups
from transformers import PreTrainedTokenizer

from .custom_datasets import ClassificationDataset

subcategories = ['alt.atheism',
                 'comp.graphics',
                 'comp.os.ms-windows.misc',
                 'comp.sys.ibm.pc.hardware',
                 'comp.sys.mac.hardware',
                 'comp.windows.x',
                 'misc.forsale',
                 'rec.autos',
                 'rec.motorcycles',
                 'rec.sport.baseball',
                 'rec.sport.hockey',
                 'sci.crypt',
                 'sci.electronics',
                 'sci.med',
                 'sci.space',
                 'soc.religion.christian',
                 'talk.politics.guns',
                 'talk.politics.mideast',
                 'talk.politics.misc',
                 'talk.religion.misc']


class TwentyNewsDataset(ClassificationDataset):

    dataset_id = "TwentyNews"
    targets = ['comp', 'rec', 'sci', 'talk']
    # ood_subtargets = comp.sys.mac.hardware,rec.sport.hockey,sci.space,talk.politics.mideast
    # ood_subtargets = comp.graphics,rec.autos,sci.med,talk.religion.misc
    # ood_subtargets = comp.windows.x,rec.motorcycles,sci.electronics,talk.politics.guns
    # ood_subtargets = comp.sys.ibm.pc.hardware,rec.sport.baseball,sci.crypt,talk.politics.misc

    def get_targets(self) -> List[str]:
        return self.targets

    def __init__(self, domain: str, split: str, data_dir: str, tokenizer: PreTrainedTokenizer, ood_targets: List[str]):
        super().__init__(domain, split, data_dir, tokenizer)
        self.ood_subtargets = ood_targets
        if domain == "id":
            # filter subcategories of shared targets with OOD, excluding OOD itself
            id_subcategories = [c for c in subcategories if any(c.startswith(t) for t in self.targets) and
                                                            c not in self.ood_subtargets]
        elif domain == "ood":
            # OOD subcategories are explicitly listed
            id_subcategories = self.ood_subtargets
        else:
            raise ValueError(domain)

        all_texts, all_targets = self.load_per_subcategories_data(id_subcategories)

        dataset_texts = self._get_in_split(all_texts)
        self.dataset_inputs = self._encode(dataset_texts)

        self.dataset_targets = self._get_in_split(all_targets)

        # deterministic mapping of the argmax ids to labels
        self._init_target_mapping(all_targets)

    def load_per_subcategories_data(self, sub_categories: Iterable[str]) -> Tuple[List[str], List[str]]:
        bunch = fetch_20newsgroups(data_home=self.data_dir, subset="all", categories=sub_categories)
        preproc_texts = [self._preproc(t) for t in bunch.data]
        preproc_targets = [self._first_order_target(t, bunch.target_names) for t in bunch.target]
        all_texts = [t for t in preproc_texts if t]
        all_targets = [y for i, y in enumerate(preproc_targets) if preproc_texts[i]]

        return self._get_in_split(all_texts), self._get_in_split(all_targets)

    def get_num_labels(self) -> int:
        return len(self.id_to_target.keys())

    @staticmethod
    def _first_order_target(target: int, all_targets: List[str]) -> str:
        return all_targets[target].split(".")[0]

    @staticmethod
    def _do_include_row(row: str) -> bool:
        return "@" not in row

    def _preproc(self, text: str) -> str:
        # remove headers of the messages, that make the system prone to over-fitting
        # look for "Lines:" header info and return given trailing number of the lines
        rows = text.split("\n")
        lines_rows = [l for l in rows if "Lines: " in l]
        if not len(lines_rows):
            # empty strings will be filtered out from samples
            return ""
        try:
            no_lines_val = int(lines_rows[0].strip().split("Lines: ")[1])
        except ValueError:
            return ""
        out_rows = [r for r in rows[-no_lines_val-1:] if TwentyNewsDataset._do_include_row(r)]

        # systematically disrupt only OOD texts:
        if self.in_or_out_domain == "ood":
            out_rows = [self._disrupt_text(r) for r in out_rows]

        return "\n".join(out_rows).strip()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {k: v[index] for k, v in self.dataset_inputs.data.items()}
        item['labels'] = self.target_to_id[self.dataset_targets[index]]
        return item

    def __len__(self) -> int:
        return len(self.dataset_inputs.data['input_ids'])
