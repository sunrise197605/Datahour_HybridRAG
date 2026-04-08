"""
Prompt template builder for the RAG pipeline.
Takes the user's question and the top-N retrieved chunks and formats them
into a single text prompt for the LLM. The prompt instructs the model to
answer using ONLY the provided context and to say "I don't know" otherwise.
"""

from typing import List

from src.types import RetrievedChunk


def build_prompt(query: str, context_chunks: List[RetrievedChunk]) -> str:
    context_blocks = []
    for idx, retrieved in enumerate(context_chunks, start=1):
        source_line = f"Source {idx}: {retrieved.chunk.title} ({retrieved.chunk.url})"
        context_blocks.append(source_line + "\n" + retrieved.chunk.text)

    context_text = "\n\n".join(context_blocks)

    prompt = (
        "You are a helpful assistant. Answer the question using only the provided context. "
        "If the answer is not present in the context, say you do not know.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context_text}\n\n"
        "Answer:"
    )
    return prompt
