from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?]) |(?<=\.)\n", text)
            if sentence.strip()
        ]
        max_len = self.max_sentences_per_chunk
        return [
            " ".join(sentences[start : start + max_len])
            for start in range(0, len(sentences), max_len)
        ]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = max(1, chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return [chunk for chunk in self._split(text, self.separators) if chunk]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        current_text = current_text.strip()
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators or remaining_separators[0] == "":
            return [
                current_text[start : start + self.chunk_size]
                for start in range(0, len(current_text), self.chunk_size)
            ]

        separator = remaining_separators[0]
        if separator not in current_text:
            return self._split(current_text, remaining_separators[1:])

        raw_parts = current_text.split(separator)
        parts = [part + separator for part in raw_parts[:-1]]
        if raw_parts[-1]:
            parts.append(raw_parts[-1])

        chunks: list[str] = []
        pending = ""
        for part in parts:
            if len(pending) + len(part) <= self.chunk_size:
                pending += part
                continue

            if pending:
                chunks.append(pending.strip())
                pending = ""

            if len(part) > self.chunk_size:
                chunks.extend(self._split(part, remaining_separators[1:]))
            else:
                pending = part

        if pending:
            chunks.append(pending.strip())
        return chunks


class HeadingChunker:
    """Chunk Markdown by headings, preserving policy/section boundaries.

    Advantage: policy and section boundaries retain their semantic context.
    Disadvantage: chunk sizes can be uneven and depend on document structure.
    """

    HEADING_PATTERN = re.compile(r"(?m)^[ \t]{0,3}#{1,6}(?!#)(?:[ \t]+|$)")
    POLICY_HEADING_PATTERN = re.compile(
        r"^[ \t]*(?P<number>\d+(?:\.\d+)*\.?)\s+(?P<title>\S.*?)\s*$"
    )
    POLICY_HEADING_PREFIXES = (
        "bán hàng",
        "các điều khoản",
        "các loại phí",
        "chi phí",
        "chính sách",
        "điều kiện",
        "điều khoản",
        "định nghĩa",
        "đối tượng",
        "hạn mức",
        "hoàn tiền",
        "hướng dẫn",
        "liên lạc",
        "lưu ý",
        "minh bạch",
        "phạm vi",
        "phân định",
        "phí",
        "quy định",
        "quyền",
        "quy trình",
        "sàn giao dịch",
        "thanh toán",
        "thời hạn",
        "tiêu chuẩn",
        "trách nhiệm",
        "tranh chấp",
    )

    def __init__(self, chunk_size: int = 500) -> None:
        self.chunk_size = max(1, chunk_size)
        self._fallback = RecursiveChunker(chunk_size=self.chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        body = self._remove_frontmatter(text).strip()
        if not body:
            return []

        heading_starts: list[int] = []
        offset = 0
        for line in body.splitlines(keepends=True):
            if self._is_heading_line(line.rstrip("\r\n")):
                heading_starts.append(offset)
            offset += len(line)

        if not heading_starts:
            return self._chunk_section(body)

        sections: list[str] = []
        preamble = body[: heading_starts[0]].strip()
        if preamble:
            sections.append(preamble)
        sections.extend(
            body[start : heading_starts[index + 1]].strip()
            if index + 1 < len(heading_starts)
            else body[start:].strip()
            for index, start in enumerate(heading_starts)
        )

        chunks: list[str] = []
        for section in sections:
            chunks.extend(self._chunk_section(section))
        return chunks

    @staticmethod
    def _remove_frontmatter(text: str) -> str:
        lines = text.splitlines(keepends=True)
        if not lines or lines[0].lstrip("\ufeff").strip() != "---":
            return text
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                return "".join(lines[index + 1 :])
        return text

    @classmethod
    def _is_heading_line(cls, line: str) -> bool:
        return bool(cls.HEADING_PATTERN.match(line)) or cls._is_policy_heading(line)

    @classmethod
    def _is_policy_heading(cls, line: str) -> bool:
        match = cls.POLICY_HEADING_PATTERN.match(line)
        if not match or "." not in match.group("number"):
            return False

        title = " ".join(match.group("title").split())
        if not title or len(title) > 160 or title.endswith((".", ";", "?", "!")):
            return False

        normalized_title = title.casefold()
        return title.isupper() or normalized_title.startswith(cls.POLICY_HEADING_PREFIXES)

    def _chunk_section(self, section: str) -> list[str]:
        section = section.strip()
        if len(section) <= self.chunk_size:
            return [section] if section else []

        heading, separator, content = section.partition("\n")
        heading = heading.rstrip()
        if not separator or not self._is_heading_line(heading):
            return self._fallback.chunk(section)

        content = content.strip()
        if not content:
            return [heading]

        content_size = max(1, self.chunk_size - len(heading) - 1)
        content_chunks = RecursiveChunker(chunk_size=content_size).chunk(content)
        return [f"{heading}\n{chunk}" for chunk in content_chunks]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    magnitude_a = math.sqrt(_dot(vec_a, vec_a))
    magnitude_b = math.sqrt(_dot(vec_b, vec_b))
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (magnitude_a * magnitude_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        safe_chunk_size = max(1, chunk_size)
        strategies = {
            "fixed_size": FixedSizeChunker(
                chunk_size=safe_chunk_size,
                overlap=min(50, safe_chunk_size - 1),
            ),
            "by_sentences": SentenceChunker(),
            "recursive": RecursiveChunker(chunk_size=safe_chunk_size),
        }

        comparison: dict[str, dict[str, object]] = {}
        for name, strategy in strategies.items():
            chunks = strategy.chunk(text)
            comparison[name] = {
                "count": len(chunks),
                "avg_length": (
                    sum(len(chunk) for chunk in chunks) / len(chunks)
                    if chunks
                    else 0.0
                ),
                "chunks": chunks,
            }
        return comparison
