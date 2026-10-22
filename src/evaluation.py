import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.embeddings import EmbeddingModel
from src.models import RetrievedChunk
from src.rag_pipeline import retrieve_relevant_chunks_for_question


@dataclass(frozen=True)
class EvalQuestion:
    """
    One question in the retrieval test set.

    expected_pages lists every page that answers the question. An empty tuple
    marks a question the notes cannot answer.
    """

    question: str
    file_name: str
    expected_pages: tuple[int, ...]

    @property
    def answerable(self) -> bool:
        return bool(self.expected_pages)


@dataclass(frozen=True)
class QuestionResult:
    """
    Retrieval outcome for one question.

    rank is the 1-based position of the first chunk from an expected page, or
    None if no relevant chunk was retrieved.
    """

    question: EvalQuestion
    retrieved: list[RetrievedChunk]
    rank: int | None

    @property
    def top_distance(self) -> float | None:
        if not self.retrieved:
            return None

        return self.retrieved[0].distance

    @property
    def relevant_distance(self) -> float | None:
        if self.rank is None:
            return None

        return self.retrieved[self.rank - 1].distance


@dataclass(frozen=True)
class ThresholdResult:
    """
    How a distance threshold trades answerable recall against rejecting
    questions the notes cannot answer.
    """

    threshold: float
    answerable_kept: float
    unanswerable_rejected: float


def load_questions(path: str | Path) -> list[EvalQuestion]:
    """
    Load a retrieval test set from a JSON file.
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    default_file_name = data.get("file_name", "")

    return [
        EvalQuestion(
            question=item["question"],
            file_name=item.get("file_name", default_file_name),
            expected_pages=tuple(item.get("pages", [])),
        )
        for item in data["questions"]
    ]


def is_relevant(chunk: RetrievedChunk, question: EvalQuestion) -> bool:
    return chunk.file_name == question.file_name and chunk.page_number in question.expected_pages


def evaluate_question(
    question: EvalQuestion,
    collection: Any,
    embedding_model: EmbeddingModel,
    top_k: int,
) -> QuestionResult:
    """
    Retrieve chunks for one question, without distance filtering, and find the
    rank of the first relevant chunk.
    """

    retrieved = retrieve_relevant_chunks_for_question(
        question=question.question,
        collection=collection,
        embedding_model=embedding_model,
        top_k=top_k,
        max_retrieval_distance=None,
    )

    rank = next(
        (index for index, chunk in enumerate(retrieved, start=1) if is_relevant(chunk, question)),
        None,
    )

    return QuestionResult(question=question, retrieved=retrieved, rank=rank)


def evaluate_questions(
    questions: list[EvalQuestion],
    collection: Any,
    embedding_model: EmbeddingModel,
    top_k: int,
) -> list[QuestionResult]:
    return [
        evaluate_question(
            question=question,
            collection=collection,
            embedding_model=embedding_model,
            top_k=top_k,
        )
        for question in questions
    ]


def _answerable(results: list[QuestionResult]) -> list[QuestionResult]:
    return [result for result in results if result.question.answerable]


def _unanswerable(results: list[QuestionResult]) -> list[QuestionResult]:
    return [result for result in results if not result.question.answerable]


def hit_rate(results: list[QuestionResult], k: int) -> float:
    """
    Share of answerable questions with a relevant chunk in the top k results.
    """

    answerable = _answerable(results)

    if not answerable:
        return 0.0

    hits = sum(1 for result in answerable if result.rank is not None and result.rank <= k)

    return hits / len(answerable)


def mean_reciprocal_rank(results: list[QuestionResult]) -> float:
    """
    Mean of 1 / rank of the first relevant chunk, counting misses as 0.
    """

    answerable = _answerable(results)

    if not answerable:
        return 0.0

    return sum(1 / result.rank for result in answerable if result.rank is not None) / len(
        answerable
    )


def threshold_tradeoff(
    results: list[QuestionResult],
    thresholds: list[float],
) -> list[ThresholdResult]:
    """
    Measure RETRIEVAL_DISTANCE_THRESHOLD candidates.

    answerable_kept is the share of answerable questions whose relevant chunk
    survives the threshold. unanswerable_rejected is the share of unanswerable
    questions where every chunk is filtered out, so the app replies that the
    answer is not in the notes without calling the chat model.
    """

    answerable = _answerable(results)
    unanswerable = _unanswerable(results)
    rows: list[ThresholdResult] = []

    for threshold in thresholds:
        kept = sum(
            1
            for result in answerable
            if result.relevant_distance is not None and result.relevant_distance <= threshold
        )
        rejected = sum(
            1
            for result in unanswerable
            if result.top_distance is None or result.top_distance > threshold
        )

        rows.append(
            ThresholdResult(
                threshold=threshold,
                answerable_kept=kept / len(answerable) if answerable else 0.0,
                unanswerable_rejected=rejected / len(unanswerable) if unanswerable else 0.0,
            )
        )

    return rows
