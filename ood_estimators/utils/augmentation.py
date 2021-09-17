from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class CyclicTranslator:

    model_name_template = "Helsinki-NLP/opus-mt-%s-%s"

    def __init__(self, src_lang: str = "en", middle_lang: str = "fr", device: str = None):
        device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"

        fwd_model = self.model_name_template % (src_lang, middle_lang)
        self.fwd_tokenizer = AutoTokenizer.from_pretrained(fwd_model)
        self.fwd_translator = AutoModelForSeq2SeqLM.from_pretrained(fwd_model).to(device)

        bwd_model = self.model_name_template % (middle_lang, src_lang)
        self.bwd_tokenizer = AutoTokenizer.from_pretrained(bwd_model)
        self.bwd_translator = AutoModelForSeq2SeqLM.from_pretrained(bwd_model).to(device)

        self.device = device

    def to(self, device: str):
        self.device = device
        self.fwd_translator = self.fwd_translator.to(device)
        self.bwd_translator = self.bwd_translator.to(device)

    def translate(self, texts: List[str]) -> List[str]:
        fwd_inputs = self.fwd_tokenizer.prepare_seq2seq_batch(texts, return_tensors="pt",
                                                              truncation=True).to(self.device)
        fwd_outputs = self.fwd_translator.generate(**fwd_inputs)
        fwd_texts = self.fwd_tokenizer.batch_decode(fwd_outputs, skip_special_tokens=True)

        bwd_inputs = self.bwd_tokenizer.prepare_seq2seq_batch(fwd_texts, return_tensors="pt",
                                                              truncation=True).to(self.device)
        bwd_outputs = self.bwd_translator.generate(**bwd_inputs)
        bwd_texts = self.bwd_tokenizer.batch_decode(bwd_outputs, skip_special_tokens=True)

        return bwd_texts
