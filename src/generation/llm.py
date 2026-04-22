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
    """Seq2Seq answer generator wrapping a Flan-T5 model.

    The generator takes a fully built prompt (question plus retrieved
    context) and returns a grounded answer string. It is a thin wrapper
    around HuggingFace AutoModelForSeq2SeqLM and AutoTokenizer for
    google/flan-t5-base, which is small enough to run on CPU without a
    GPU, yet instruction-tuned well enough to follow the "answer only
    from the provided context" prompt reliably. Generation is done with
    greedy decoding (do_sample=False) so that answers are reproducible
    across runs, which matters for evaluation harnesses and demos.
    The model is loaded once via load() and reused for every query to
    avoid repeated cold starts. Swap model_name to flan-t5-large or a
    Mistral chat model to trade latency for higher answer quality.
    """

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
