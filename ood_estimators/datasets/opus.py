import os
from typing import Iterator, Tuple, List, Optional

from torch.utils.data.dataset import T_co
from transformers import PreTrainedTokenizer

from .custom_datasets import TranslationDataset

SRC_LANG = "no"
TGT_LANG = "de"

# in OPUS, lang pair identifier is sorted alphabetically
ordered_lang_pair = tuple(sorted([SRC_LANG, TGT_LANG]))

# these can be easily exchanged with links to other domains and languages in MOSES format
# see a list for your language pair in https://opus.nlpl.eu/
OPUS_RESOURCES_URLS = {
    "WikiMatrix": "https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/%s-%s.txt.zip"
                  % ordered_lang_pair,  # In-distribution
    "OpenSubtitles": "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/%s-%s.txt.zip"
                     % ordered_lang_pair,  # val Out-of-distribution
    "Bible": "https://object.pouta.csc.fi/OPUS-bible-uedin/v1/moses/%s-%s.txt.zip"
             % ordered_lang_pair,  # test Out-of-distribution
}

# we make sure that no duplicates are loaded among the resources
loaded_srcs = set()


class OPUSDataset(TranslationDataset):
    dataset_id = "OPUS"

    ID_DOMAIN = "WikiMatrix"
    OOD_TEST_DOMAIN = "Bible"
    OOD_DOMAIN = "OpenSubtitles"

    def __init__(self, domain: str, split: str, data_dir: str,
                 tokenizer: PreTrainedTokenizer, firstn: Optional[int] = 100):
        super().__init__(domain, split, data_dir, tokenizer)
        if domain == "id":
            source_texts, target_texts = self._load_translation_pairs(self.ID_DOMAIN, firstn)
        elif domain == "ood":
            source_texts, target_texts = self._load_translation_pairs(self.OOD_DOMAIN, firstn)
        else:
            raise ValueError(domain)

        self.source = source_texts
        self.target = target_texts

        self.data_dir = data_dir

        # this will not fit the memory
        # self.dataset_inputs = self._encode_pairs(self.source, self.target, tokenizer)

    @staticmethod
    def _preproc(text: str):
        return text.strip()

    @staticmethod
    def _deduplicate(src_texts: List[str], tgt_texts: List[str]) -> Tuple[List[str], List[str]]:
        out_src_texts = []
        out_tgt_texts = []
        for src_text, tgt_text in zip(src_texts, tgt_texts):
            if src_text not in loaded_srcs:
                out_src_texts.append(src_text)
                out_tgt_texts.append(tgt_text)
                loaded_srcs.add(src_text)

        return out_src_texts, out_tgt_texts

    def _load_translation_pairs(self, domain_label: str, firstn: int) -> Tuple[List[str], List[str]]:
        src_file, tgt_file = self._maybe_download_unzip(domain_label)
        with open(src_file, "r") as f:
            src_lines = [self._preproc(l) for l in f.readlines()]
        with open(tgt_file, "r") as f:
            tgt_lines = [self._preproc(l) for l in f.readlines()]

        src_lines = self._get_in_split(src_lines)
        tgt_lines = self._get_in_split(tgt_lines)

        src_lines, tgt_lines = self._deduplicate(src_lines, tgt_lines)

        if firstn is not None:
            src_lines = src_lines[:firstn]
            tgt_lines = tgt_lines[:firstn]

        if self.in_or_out_domain == "ood":
            src_lines = [self._disrupt_text(l) for l in src_lines]

        return src_lines, tgt_lines

    def _maybe_download_unzip(self, domain_label: str) -> Tuple[str, str]:
        src_suffix = "%s-%s.%s" % (ordered_lang_pair[0], ordered_lang_pair[1], SRC_LANG)
        tgt_suffix = "%s-%s.%s" % (ordered_lang_pair[0], ordered_lang_pair[1], TGT_LANG)
        from zipfile import ZipFile
        from urllib.request import urlopen
        from io import BytesIO

        out_srcs = [os.path.join(self.data_dir, fpath) for fpath in os.listdir(self.data_dir)
                    if domain_label in fpath and src_suffix in fpath]
        out_tgts = [os.path.join(self.data_dir, fpath) for fpath in os.listdir(self.data_dir)
                    if domain_label in fpath and tgt_suffix in fpath]

        # resources are not yet downloaded
        if not out_srcs or not out_tgts:
            url = OPUS_RESOURCES_URLS[domain_label]
            resp = urlopen(url)
            with ZipFile(BytesIO(resp.read())) as zipfile:
                files_in_zip = zipfile.NameToInfo.keys()
                src_zip_path = [zipfile for zipfile in files_in_zip if src_suffix in zipfile][0]
                tgt_zip_path = [zipfile for zipfile in files_in_zip if tgt_suffix in zipfile][0]
                for cached_f in [src_zip_path, tgt_zip_path]:
                    zipfile.extract(cached_f, path=self.data_dir)
                    assert os.path.exists(os.path.join(self.data_dir, cached_f))

            out_src_f = os.path.join(self.data_dir, src_zip_path)
            out_tgt_f = os.path.join(self.data_dir, tgt_zip_path)
        else:
            out_src_f = out_srcs[0]
            out_tgt_f = out_tgts[0]

        return out_src_f, out_tgt_f

    def __getitem__(self, index: int) -> T_co:
        encoded_batch = self._encode_pairs(self.source[index], self.target[index], self.tokenizer)
        return {k: v[0] for k, v in encoded_batch.items()}

    def __len__(self) -> int:
        return len(self.source)
