from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Insufficient context: no relevant information was found."

        context_parts: list[str] = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata") or {}
            source_id = metadata.get("doc_id") or result.get("id", "unknown")
            context_parts.append(
                f"[{index}] Source: {source_id}\n{result.get('content', '')}"
            )

        context = "\n\n".join(context_parts)
        prompt = (
            "Answer the question using only the supplied context. "
            "Do not invent unsupported information. If the context is insufficient, "
            "say so clearly. Cite supporting chunks using [1], [2], etc.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return self.llm_fn(prompt)
