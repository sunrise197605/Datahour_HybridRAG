"""
Prompt template builder for the RAG pipeline.
Takes the user's question and the top-N retrieved chunks and formats them
into a single text prompt for the LLM. The prompt instructs the model to
answer using ONLY the provided context and to say "I don't know" otherwise.
"""

from typing import List

from src.types import RetrievedChunk


def build_prompt(query: str, context_chunks: List[RetrievedChunk]) -> str:
    """Assemble the final LLM prompt from a query and retrieved context.

    The prompt is structured in three parts: an instruction, the question,
    and the context. The instruction tells the model to answer strictly
    from the provided context, to address every part of a multi-part
    question, and to include concrete facts like dates, locations, and
    numbers. Each retrieved chunk is prefixed with a numbered source
    header (title plus URL) so the model can be gently steered toward
    grounded phrasing and so the downstream UI can map each claim back
    to a clickable citation. The instruction also tells the model to
    admit when it does not know, which is the main guardrail against
    hallucination for open-domain questions.
    """
    context_blocks = []
    for idx, retrieved in enumerate(context_chunks, start=1):
        source_line = f"Source {idx}: {retrieved.chunk.title} ({retrieved.chunk.url})"
        context_blocks.append(source_line + "\n" + retrieved.chunk.text)

    context_text = "\n\n".join(context_blocks)

    prompt = (
        "You are a helpful assistant. Answer the question using only the provided context. "
        "If the question has multiple parts, address every part in one complete sentence. "
        "Include all relevant facts such as locations, dates, years, and numbers. "
        "If the answer is not present in the context, say you do not know.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context_text}\n\n"
        "Complete answer:"
    )
    return prompt
