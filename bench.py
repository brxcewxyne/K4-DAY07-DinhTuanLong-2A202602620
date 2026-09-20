from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from src.chunking import ChunkingStrategyComparator, HeadingChunker
from src.embeddings import LocalEmbedder
from src.models import Document
from src.store import EmbeddingStore


ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "data" / "shopee-bao hanh"
OUTPUT_PATH = ROOT / "ket_qua_benchmark.txt"
SAMPLE_DOC_IDS = (
    "return-refund-policy",
    "shopee-mall-seller",
    "warranty-general-seller",
)

BENCHMARK_QUESTIONS = [
    {
        "id": "Q1",
        "query": "Người Mua có bao lâu để gửi yêu cầu trả hàng/hoàn tiền sau khi đơn được giao thành công?",
        "metadata_filter": None,
        "gold_answer": "15 ngày lịch đối với sản phẩm thông thường; 24 giờ đối với thực phẩm tươi sống và đông lạnh.",
        "answer_markers": ["15 (mười lăm) ngày", "24 (hai mươi bốn) giờ đồng hồ"],
    },
    {
        "id": "Q2",
        "query": "Sau khi yêu cầu trả hàng/hoàn tiền Shopee Mall được chấp thuận, Người Mua có bao lâu để gửi trả sản phẩm?",
        "metadata_filter": {"audience": "buyer"},
        "gold_answer": "6 ngày lịch kể từ khi yêu cầu được chấp thuận.",
        "answer_markers": ["06 (sáu) ngày lịch kể từ ngày yêu cầu trả hàng/hoàn tiền được chấp thuận"],
    },
    {
        "id": "Q3",
        "query": "Khi Shopee yêu cầu bằng chứng cho một yêu cầu trả hàng/hoàn tiền Shopee Mall, Người Bán phải cung cấp trong bao lâu?",
        "metadata_filter": {"audience": "seller"},
        "gold_answer": "Tối đa 24 giờ kể từ khi nhận được yêu cầu của Shopee.",
        "answer_markers": [
            "có trách nhiệm cung cấp các bằng chứng liên quan trong vòng tối đa 24 (hai mươi bốn) giờ",
            "trong vòng tối đa 24 (hai mươi bốn) giờ kể từ khi nhận được yêu cầu của Shopee",
        ],
    },
    {
        "id": "Q4",
        "query": "Ai chịu trách nhiệm tiếp nhận bảo hành sản phẩm cho Người Mua trên Shopee?",
        "metadata_filter": {"audience": "seller"},
        "gold_answer": "Người Bán chịu trách nhiệm tiếp nhận bảo hành theo chính sách đã công bố; Shopee không trực tiếp thực hiện nghĩa vụ bảo hành, trừ sản phẩm do chính Shopee đăng bán.",
        "answer_markers": [
            "Người Bán có trách nhiệm tiếp nhận bảo hành",
            "Shopee sẽ không chịu trách nhiệm trong việc bảo hành",
            "Shopee không phải là bên thực hiện nghĩa vụ bảo hành",
        ],
    },
    {
        "id": "Q5",
        "query": "Đối với tranh chấp không phải khiếu nại trả hàng/hoàn tiền, Shopee đưa ra hướng giải quyết trong bao lâu sau khi nhận đủ tài liệu?",
        "metadata_filter": {"audience": "both"},
        "gold_answer": "Trong vòng 07 ngày làm việc kể từ ngày nhận đủ thông tin/tài liệu; trường hợp phức tạp có thể kéo dài hơn.",
        "answer_markers": [
            "trong vòng 07 ngày làm việc kể từ ngày nhận được đầy đủ",
            "thời hạn giải quyết sẽ được kéo dài hơn",
        ],
    },
]

AB_TEST = {
    "query": "Khi Shopee yêu cầu cung cấp bằng chứng cho yêu cầu trả hàng/hoàn tiền Shopee Mall thì phải cung cấp trong bao lâu?",
    "metadata_filter": {"audience": "seller"},
    "answer_markers": BENCHMARK_QUESTIONS[2]["answer_markers"],
}


class Reporter:
    """Write the same benchmark output to the terminal and an export buffer."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def line(self, value: object = "") -> None:
        text = str(value)
        print(text)
        self.lines.append(text)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Return simple YAML-style scalar metadata and the Markdown body."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].lstrip("\ufeff").strip() != "---":
        return {}, text
    closing_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing_index is None:
        return {}, text
    metadata: dict[str, str] = {}
    for line in lines[1:closing_index]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        metadata[key.strip()] = value
    return metadata, "".join(lines[closing_index + 1 :]).strip()


def print_baseline(sample_size: int = 3) -> None:
    """Optionally print baseline chunk statistics for a few corpus files."""
    comparator = ChunkingStrategyComparator()
    for path in sorted(CORPUS_DIR.glob("*.md"))[:sample_size]:
        _, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        print(f"\nBaseline: {path.name}")
        for name, stats in comparator.compare(body).items():
            print(f"  {name:12} count={stats['count']:3} avg_length={stats['avg_length']:.1f}")


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _matching_marker(content: str, markers: list[str]) -> str | None:
    normalized_content = _normalize(content)
    return next((marker for marker in markers if _normalize(marker) in normalized_content), None)


def _first_answer_rank(results: list[dict[str, Any]], markers: list[str]) -> int | None:
    return next(
        (rank for rank, result in enumerate(results, start=1) if _matching_marker(result["content"], markers)),
        None,
    )


def _query_score(answer_rank: int | None) -> int:
    if answer_rank == 1:
        return 2
    if answer_rank in (2, 3):
        return 1
    return 0


def _print_results(reporter: Reporter, results: list[dict[str, Any]], markers: list[str]) -> None:
    if not results:
        reporter.line("  No matching results.")
        return
    for rank, result in enumerate(results, start=1):
        marker = _matching_marker(result["content"], markers)
        preview = " ".join(result["content"].split())[:180]
        reporter.line(
            f"  {rank}. score={result['score']:.4f} "
            f"doc_id={result['metadata']['doc_id']} "
            f"chunk_id={result['id']} marker={'YES' if marker else 'no'}"
        )
        reporter.line(f"     {preview}")
        if marker:
            reporter.line(f"     matched: {marker}")


def _metadata_matches(
    metadata: dict[str, Any], metadata_filter: dict[str, Any] | None
) -> bool:
    return not metadata_filter or all(
        metadata.get(key) == value for key, value in metadata_filter.items()
    )


def _describe_ab_result(
    unfiltered: list[dict[str, Any]], filtered: list[dict[str, Any]], markers: list[str]
) -> str:
    rank_without = _first_answer_rank(unfiltered, markers)
    rank_with = _first_answer_rank(filtered, markers)
    ids_without = [result["id"] for result in unfiltered]
    ids_with = [result["id"] for result in filtered]
    if rank_without is None and rank_with is not None:
        return f"Filtering moved an answer-containing chunk into top-3 at rank {rank_with}."
    if rank_without is not None and rank_with is None:
        return f"Filtering removed the answer-containing chunk previously at rank {rank_without}; it did not help this query."
    if rank_without is not None and rank_with is not None and rank_with < rank_without:
        return f"Filtering improved the answer-containing chunk from rank {rank_without} to rank {rank_with}."
    if ids_without == ids_with:
        return "Filtered and unfiltered top-3 results were effectively identical."
    if rank_without is not None and rank_with == rank_without:
        return f"Filtering changed results but left the answer-containing chunk at rank {rank_with}."
    return "Filtering changed the top-3 but did not retrieve an answer-containing chunk in either run."


def _failure_analysis(
    evaluations: list[dict[str, Any]], documents: list[Document]
) -> tuple[dict[str, Any], str, str]:
    failed = [evaluation for evaluation in evaluations if evaluation["query_score"] == 0]
    candidates = failed or sorted(evaluations, key=lambda item: item["query_score"])
    for evaluation in candidates:
        benchmark = evaluation["benchmark"]
        answer_documents = [
            document
            for document in documents
            if _matching_marker(document.content, benchmark["answer_markers"])
        ]
        allowed_answers = [
            document
            for document in answer_documents
            if _metadata_matches(document.metadata, benchmark["metadata_filter"])
        ]
        if answer_documents and not allowed_answers:
            reason = (
                f"The metadata filter {benchmark['metadata_filter']} excluded all answer-containing "
                f"chunks, including {answer_documents[0].id}."
            )
            suggestion = (
                "Improve audience metadata for mixed-audience policy sections by labeling the "
                "seller-facing section as seller/both or supporting multiple audiences."
            )
            return evaluation, reason, suggestion

    evaluation = candidates[0]
    benchmark = evaluation["benchmark"]
    answer_documents = [
        document
        for document in documents
        if _matching_marker(document.content, benchmark["answer_markers"])
        and _metadata_matches(document.metadata, benchmark["metadata_filter"])
    ]
    retrieved_doc_ids = {result["metadata"]["doc_id"] for result in evaluation["results"]}
    answer_doc_ids = {document.metadata["doc_id"] for document in answer_documents}
    if retrieved_doc_ids & answer_doc_ids:
        reason = "The correct source document was retrieved, but answer-less sections ranked above the answer chunk."
    else:
        reason = "Semantically related but answer-less chunks ranked above the available answer chunk."
    suggestion = "Add hybrid lexical/semantic retrieval or reranking that rewards exact time and policy phrases."
    return evaluation, reason, suggestion


def run_benchmark() -> None:
    reporter = Reporter()
    paths = sorted(CORPUS_DIR.glob("*.md"))
    try:
        embedder = LocalEmbedder()
    except ModuleNotFoundError as error:
        missing = error.name or "sentence-transformers"
        raise SystemExit(
            "Local semantic embeddings are required for this benchmark. "
            f"Missing dependency: {missing}. Install requirements-local.txt and run python bench.py again."
        ) from error

    # Change only this line to benchmark another chunking strategy.
    chunker = HeadingChunker(chunk_size=800)
    documents: list[Document] = []
    chunk_counts: dict[str, int] = {}
    chunks_by_source: dict[str, list[str]] = {}
    for path in paths:
        frontmatter, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        chunks = chunker.chunk(body)
        chunk_counts[path.stem] = len(chunks)
        chunks_by_source[path.stem] = chunks
        documents.extend(
            Document(
                id=f"{path.stem}#{index}",
                content=chunk,
                metadata={**frontmatter, "doc_id": path.stem},
            )
            for index, chunk in enumerate(chunks)
        )

    store = EmbeddingStore(collection_name="shopee_heading_benchmark", embedding_fn=embedder)
    store.add_documents(documents)
    reporter.line(f"Embedding backend: {embedder._backend_name}")
    reporter.line(f"Chunk strategy: {chunker.__class__.__name__} (chunk_size={chunker.chunk_size})")
    reporter.line(f"Source files: {len(paths)}")
    reporter.line(f"Total chunks loaded: {store.get_collection_size()}")
    reporter.line("Chunks per source file:")
    for doc_id, count in chunk_counts.items():
        reporter.line(f"  {doc_id}: {count}")

    reporter.line("\nSample policy-heading chunks:")
    for doc_id in SAMPLE_DOC_IDS:
        reporter.line(f"  {doc_id}:")
        samples: list[tuple[int, str]] = []
        sampled_headings: set[str] = set()
        for index, chunk in enumerate(chunks_by_source[doc_id]):
            heading = chunk.splitlines()[0]
            if chunk.startswith("#") or heading in sampled_headings:
                continue
            samples.append((index, chunk))
            sampled_headings.add(heading)
            if len(samples) == 3:
                break
        for index, chunk in samples:
            reporter.line(f"    {doc_id}#{index}: {' '.join(chunk.split())[:200]}")

    reporter.line("\n=== Five-query retrieval benchmark ===")
    evaluations: list[dict[str, Any]] = []
    total_score = 0
    relevant_count = 0
    for benchmark in BENCHMARK_QUESTIONS:
        results = store.search_with_filter(
            benchmark["query"], top_k=3, metadata_filter=benchmark["metadata_filter"]
        )
        answer_rank = _first_answer_rank(results, benchmark["answer_markers"])
        query_score = _query_score(answer_rank)
        total_score += query_score
        relevant_count += int(answer_rank is not None)
        evaluations.append(
            {"benchmark": benchmark, "results": results, "answer_rank": answer_rank, "query_score": query_score}
        )
        reporter.line(f"\n{benchmark['id']}: {benchmark['query']}")
        reporter.line(f"Filter: {benchmark['metadata_filter']}")
        reporter.line(f"Gold answer: {benchmark['gold_answer']}")
        _print_results(reporter, results, benchmark["answer_markers"])
        reporter.line(f"Query score: {query_score} / 2")
        reporter.line(
            "Answer-containing chunk in top-3: "
            + (f"YES (rank {answer_rank})" if answer_rank else "NO")
        )

    reporter.line(f"\nTotal benchmark score: {total_score} / 10")
    reporter.line(f"Queries with answer-containing chunk in top-3: {relevant_count} / 5")

    reporter.line("\n=== Metadata A/B test ===")
    reporter.line(f"Query: {AB_TEST['query']}")
    unfiltered = store.search(AB_TEST["query"], top_k=3)
    reporter.line("\nA. WITHOUT metadata filter")
    _print_results(reporter, unfiltered, AB_TEST["answer_markers"])
    filtered = store.search_with_filter(
        AB_TEST["query"], top_k=3, metadata_filter=AB_TEST["metadata_filter"]
    )
    reporter.line(f"\nB. WITH metadata filter {AB_TEST['metadata_filter']}")
    _print_results(reporter, filtered, AB_TEST["answer_markers"])
    ab_summary = _describe_ab_result(unfiltered, filtered, AB_TEST["answer_markers"])
    reporter.line(f"A/B summary: {ab_summary}")

    failure, failure_reason, improvement = _failure_analysis(evaluations, documents)
    failed_benchmark = failure["benchmark"]
    actual_ids = ", ".join(result["id"] for result in failure["results"])
    reporter.line("\n=== Failure analysis ===")
    reporter.line(f"Query: {failed_benchmark['query']}")
    reporter.line(f"Expected: {failed_benchmark['gold_answer']}")
    reporter.line(f"Top-3 returned: {actual_ids or 'no results'}")
    reporter.line(f"Likely reason: {failure_reason}")
    reporter.line(f"Improvement: {improvement}")

    strongest = max(evaluations, key=lambda item: item["query_score"])
    weakest = min(evaluations, key=lambda item: item["query_score"])
    reporter.line("\n=== Report-ready summary ===")
    reporter.line("Strategy: HeadingChunker")
    reporter.line(f"Embedding backend: {embedder._backend_name}")
    reporter.line(f"Total chunks: {store.get_collection_size()}")
    reporter.line(f"Relevant answer chunk in top-3: {relevant_count}/5")
    reporter.line(f"Retrieval score: {total_score}/10")
    reporter.line(f"Strongest query: {strongest['benchmark']['id']}")
    reporter.line(f"Weakest query: {weakest['benchmark']['id']}")
    reporter.line(f"Metadata A/B result: {ab_summary}")
    reporter.line(f"Failure insight: {failure_reason}")
    reporter.line(
        "Agent evaluation: skipped because no configured real llm_fn is available; "
        "agent-answer correctness must be evaluated separately."
    )
    reporter.line(f"Exported results: {OUTPUT_PATH}")
    reporter.save(OUTPUT_PATH)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_benchmark()
