"""
Wrapper for Mistral-7B-Instruct causal LM, used as the LLM-as-Judge evaluator.
Formats a list of ChatMessage objects (system/user/assistant roles) into
the Mistral instruct prompt format and generates a response. This is only
needed for the LLM-judge evaluation and question generation steps — the main
RAG pipeline uses the lighter Flan-T5 model (see llm.py) instead.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class MistralChat:
    """Wrapper for Mistral Instruct-style models (e.g., Mistral-7B-Instruct).
    Formats messages into the [SYSTEM]/[USER]/[ASSISTANT] prompt format."""
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
            device_map="auto" if self.device.startswith("cuda") else None,
        )
        if not self.device.startswith("cuda"):
            self.model.to(self.device)

    def _format_chat(self, messages: List[ChatMessage]) -> str:
        parts: List[str] = []
        for message in messages:
            if message.role == "system":
                parts.append(f"[SYSTEM]\n{message.content}\n")
            elif message.role == "user":
                parts.append(f"[USER]\n{message.content}\n")
            else:
                parts.append(f"[ASSISTANT]\n{message.content}\n")
        parts.append("[ASSISTANT]\n")
        return "\n".join(parts)

    def generate(self, messages: List[ChatMessage], max_new_tokens: int = 512, temperature: float = 0.2) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load().")

        prompt = self._format_chat(messages)
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        marker = "[ASSISTANT]"
        if marker in decoded:
            return decoded.split(marker)[-1].strip()
        return decoded.strip()
