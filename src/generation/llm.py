"""
LLM answer generator using HuggingFace Flan-T5.
Loads a Seq2Seq model (default: google/flan-t5-base) and its tokenizer,
then generates answers from a prompt containing the question and retrieved
context. Runs on CPU by default so no GPU is required for the demo.
The generate() method uses greedy decoding (do_sample=False) for reproducibility.
"""

from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class LLMGenerator:
    def __init__(self, model_name: str = "google/flan-t5-base", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model: Optional[AutoModelForSeq2SeqLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLMGenerator is not loaded. Call load() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
