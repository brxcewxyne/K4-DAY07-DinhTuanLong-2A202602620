"""Agent evaluation with Gemini answer generation.

Retrieval pipeline (unchanged from bench.py):
  corpus -> HeadingChunker(chunk_size=800) -> LocalEmbedder
  (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
  -> EmbeddingStore.search_with_filter (top-3, benchmark metadata filters)

Generation backend: Gemini API (google-genai SDK), used ONLY for the final
answer. Set the key before running (Windows CMD)::

    set GEMINI_API_KEY=YOUR_KEY
    .venv\\Scripts\\python.exe agent_eval.py

Optional overrides:
  GEMINI_MODEL  Gemini model id (default: gemini-2.5-flash)

KnowledgeBaseAgent itself is left untouched: it calls store.search(), so each
query runs through a FilteredSearchStore that exposes the benchmark
metadata_filter via the Agent's search API. The grounded prompt format is the
one built inside KnowledgeBaseAgent.answer().
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from bench import (
    BENCHMARK_QUESTIONS,
    CORPUS_DIR,
    _first_answer_rank,
    parse_frontmatter,
)
from src.agent import KnowledgeBaseAgent
from src.chunking import HeadingChunker
from src.embeddings import LocalEmbedder
from src.models import Document
from src.store import EmbeddingStore


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OUTPUT_PATH = Path(__file__).resolve().parent / "agent_results.txt"

INSUFFICIENT_SENTENCE = "Ngữ cảnh được cung cấp không đủ để trả lời câu hỏi."

SYSTEM_INSTRUCTION = (
    "Bạn là trợ lý hỏi đáp có căn cứ. Trả lời ngắn gọn bằng tiếng Việt, "
    "chỉ dùng thông tin được nêu trực tiếp trong ngữ cảnh được cung cấp, "
    "không dùng kiến thức bên ngoài. Trích dẫn chunk hỗ trợ dạng [1], [2], [3]. "
    "Không bộc lộ chuỗi suy luận. Nếu ngữ cảnh không trực tiếp chứa câu trả lời, "
    f"chỉ trả lời đúng một câu: '{INSUFFICIENT_SENTENCE}'"
)


class FilteredSearchStore:
    """Expose benchmark-filtered retrieval through the Agent's search API."""

    def __init__(
        self, store: EmbeddingStore, metadata_filter: dict[str, Any] | None
    ) -> None:
        self.store = store
        self.metadata_filter = metadata_filter

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.store.search_with_filter(
            query, top_k=top_k, metadata_filter=self.metadata_filter
        )


class Reporter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def line(self, value: object = "") -> None:
        rendered = str(value)
        print(rendered)
        self.lines.append(rendered)

    def save(self) -> None:
        OUTPUT_PATH.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise SystemExit(
            "GEMINI_API_KEY or GOOGLE_API_KEY is required. "
            "Set it before running (Windows CMD): set GEMINI_API_KEY=YOUR_KEY"
        )
    return key


class _GeminiLLM:
    """Deterministic Gemini answer generator (google-genai SDK)."""

    def __init__(self, model: str) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as error:
            raise SystemExit(
                "The google-genai package is required for Agent evaluation. "
                "Install it with: .venv\\Scripts\\python.exe -m pip install google-genai"
            ) from error
        self._types = types
        self.model = model
        self.client = genai.Client(api_key=_api_key())

    def __call__(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.0,
                max_output_tokens=512,
            ),
        )
        answer = (response.text or "").strip()
        if not answer:
            raise RuntimeError("Gemini returned an empty answer.")
        return answer


def _load_documents() -> list[Document]:
    chunker = HeadingChunker(chunk_size=800)
    documents: list[Document] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        frontmatter, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        documents.extend(
            Document(
                id=f"{path.stem}#{index}",
                content=chunk,
                metadata={**frontmatter, "doc_id": path.stem},
            )
            for index, chunk in enumerate(chunker.chunk(body))
        )
    return documents


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _contains_number(text: str, number: int) -> bool:
    return bool(re.search(rf"(?<!\d)0?{number}(?!\d)", text))


def _is_insufficient_answer(answer: str) -> bool:
    text = _normalized(answer)
    return any(
        phrase in text
        for phrase in (
            "không đủ thông tin",
            "không có đủ thông tin",
            "ngữ cảnh không đủ",
            "ngữ cảnh được cung cấp không đủ",
            "không tìm thấy",
            "insufficient context",
        )
    )


def _answer_is_correct(query_id: str, answer: str) -> bool:
    """Evaluation-only check of the Agent answer against the gold answer."""
    text = _normalized(answer)
    checks = {
        "Q1": _contains_number(text, 15)
        and _contains_number(text, 24)
        and "giờ" in text,
        "Q2": _contains_number(text, 6) and "ngày" in text,
        "Q3": _contains_number(text, 24) and "giờ" in text,
        # Q4 has no answer-bearing chunk in top-3: the only acceptable
        # answer is an explicit insufficient-context statement.
        "Q4": _is_insufficient_answer(answer),
        "Q5": _contains_number(text, 7)
        and "ngày làm việc" in text
        and "phức tạp" in text
        and "kéo dài" in text,
    }
    return checks[query_id]


def _rubric_score(evidence_rank: int | None, answer_correct: bool) -> int:
    """Official docs/SCORING.md rubric per query (2 / 1 / 0)."""
    if evidence_rank is None:
        return 0
    if evidence_rank == 1 and answer_correct:
        return 2
    return 1


def _retrieval_proxy_score(evidence_rank: int | None) -> int:
    """Retrieval-only diagnostic score (same rank rule as bench.py)."""
    if evidence_rank == 1:
        return 2
    if evidence_rank in (2, 3):
        return 1
    return 0


def run_evaluation() -> None:
    llm = _GeminiLLM(GEMINI_MODEL)

    documents = _load_documents()
    embedder = LocalEmbedder()
    store = EmbeddingStore(
        collection_name="shopee_heading_agent_evaluation", embedding_fn=embedder
    )
    store.add_documents(documents)

    reporter = Reporter()
    reporter.line("Generation backend: Gemini API (google-genai SDK)")
    reporter.line(f"Gemini model: {GEMINI_MODEL}")
    reporter.line(f"Embedding backend: {embedder._backend_name}")
    reporter.line("Chunk strategy: HeadingChunker (chunk_size=800)")
    reporter.line(f"Corpus files: {len(sorted(CORPUS_DIR.glob('*.md')))}")
    reporter.line(f"Total chunks: {store.get_collection_size()}")

    total_score = 0
    retrieval_proxy_total = 0
    evaluations: list[dict[str, Any]] = []
    for benchmark in BENCHMARK_QUESTIONS:
        query_id = benchmark["id"]
        query = benchmark["query"]
        metadata_filter = benchmark["metadata_filter"]
        results = store.search_with_filter(
            query, top_k=3, metadata_filter=metadata_filter
        )
        evidence_rank = _first_answer_rank(results, benchmark["answer_markers"])

        # Same grounded prompt as KnowledgeBaseAgent; retrieval honors the
        # benchmark metadata filter via the wrapper (KnowledgeBaseAgent
        # itself is unchanged).
        filtered_store = FilteredSearchStore(store, metadata_filter)
        agent = KnowledgeBaseAgent(store=filtered_store, llm_fn=llm)
        answer = agent.answer(query, top_k=3)

        answer_correct = _answer_is_correct(query_id, answer)
        insufficient = _is_insufficient_answer(answer)
        if evidence_rank is None:
            grounding = "GROUNDED_INSUFFICIENT" if insufficient else "UNSUPPORTED"
            supported = insufficient
        elif answer_correct:
            grounding = "GROUNDED_CORRECT"
            supported = True
        else:
            grounding = "INCORRECT"
            supported = False

        score = _rubric_score(evidence_rank, answer_correct)
        retrieval_proxy_total += _retrieval_proxy_score(evidence_rank)
        total_score += score
        evaluations.append(
            {
                "id": query_id,
                "answer": answer,
                "evidence_rank": evidence_rank,
                "supported": supported,
                "correct": answer_correct,
                "grounding": grounding,
                "score": score,
            }
        )

        reporter.line(f"\n=== {query_id} ===")
        reporter.line(f"Query: {query}")
        reporter.line(f"Metadata filter: {metadata_filter}")
        for rank, result in enumerate(results, start=1):
            preview = " ".join(result["content"].split())[:220]
            reporter.line(
                f"{rank}. chunk_id={result['id']} "
                f"doc_id={result['metadata']['doc_id']} score={result['score']:.4f}"
            )
            reporter.line(f"   {preview}")
        reporter.line(
            "Gold-answer evidence in top-3: "
            + (f"YES (rank {evidence_rank})" if evidence_rank else "NO")
        )
        reporter.line(f"Agent answer: {answer}")
        reporter.line(f"Gold answer: {benchmark['gold_answer']}")
        reporter.line(f"Grounding verdict: {grounding}")
        reporter.line(f"Supported by retrieved context: {'YES' if supported else 'NO'}")
        reporter.line(f"Agent answer correct: {'YES' if answer_correct else 'NO'}")
        reporter.line(f"Official rubric score: {score} / 2")

    reporter.line("\n=== Official Section 5 score (docs/SCORING.md) ===")
    for evaluation in evaluations:
        rank = evaluation["evidence_rank"]
        if rank is None:
            reason = "no answer-containing chunk in top-3"
        elif evaluation["score"] == 2:
            reason = "answer evidence at rank 1 and Agent answer correct"
        elif rank != 1:
            reason = f"answer evidence at rank {rank}, so the rubric caps this at 1"
        else:
            reason = "Agent answer was incomplete or incorrect"
        reporter.line(f"{evaluation['id']}: {evaluation['score']} / 2 — {reason}")
    reporter.line(f"Total official Agent-evaluated Section 5 score: {total_score} / 10")
    reporter.line(
        f"Retrieval-only diagnostic score (rank proxy, no LLM): "
        f"{retrieval_proxy_total} / 10"
    )
    reporter.save()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_evaluation()
