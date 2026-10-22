import json

import pytest

from src.evaluation import (
    EvalQuestion,
    QuestionResult,
    evaluate_questions,
    hit_rate,
    load_questions,
    mean_reciprocal_rank,
    threshold_tradeoff,
)
from src.models import RetrievedChunk
from src.vector_store import get_chroma_client, get_or_create_collection, upsert_embedded_chunks
from tests.helpers import KeywordFakeEmbeddingModel, make_embedded_chunk


def make_chunk(page_number: int, distance: float, file_name: str = "notes.pdf") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk_{page_number}",
        file_name=file_name,
        page_number=page_number,
        text="text",
        distance=distance,
    )


def make_result(expected_pages: tuple[int, ...], retrieved: list[RetrievedChunk]) -> QuestionResult:
    question = EvalQuestion(question="q", file_name="notes.pdf", expected_pages=expected_pages)
    rank = next(
        (
            index
            for index, chunk in enumerate(retrieved, start=1)
            if chunk.page_number in expected_pages
        ),
        None,
    )
    return QuestionResult(question=question, retrieved=retrieved, rank=rank)


@pytest.fixture
def results() -> list[QuestionResult]:
    return [
        make_result((1,), [make_chunk(1, 0.2), make_chunk(2, 0.5)]),
        make_result((2,), [make_chunk(1, 0.3), make_chunk(2, 0.6)]),
        make_result((3,), [make_chunk(1, 0.4), make_chunk(2, 0.7)]),
        make_result((), [make_chunk(1, 0.9)]),
        make_result((), [make_chunk(2, 0.45)]),
    ]


def test_load_questions_reads_pages_and_default_file_name(tmp_path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            {
                "file_name": "notes.pdf",
                "questions": [
                    {"question": "What is recursion?", "pages": [2, 3]},
                    {"question": "Capital of Finland?", "pages": []},
                ],
            }
        ),
        encoding="utf-8",
    )

    questions = load_questions(path)

    assert questions[0] == EvalQuestion("What is recursion?", "notes.pdf", (2, 3))
    assert questions[0].answerable
    assert not questions[1].answerable


def test_hit_rate_counts_only_answerable_questions(results) -> None:
    assert hit_rate(results, 1) == pytest.approx(1 / 3)
    assert hit_rate(results, 2) == pytest.approx(2 / 3)


def test_mean_reciprocal_rank_counts_misses_as_zero(results) -> None:
    assert mean_reciprocal_rank(results) == pytest.approx((1 + 0.5 + 0) / 3)


def test_metrics_are_zero_without_answerable_questions() -> None:
    results = [make_result((), [make_chunk(1, 0.9)])]

    assert hit_rate(results, 5) == 0.0
    assert mean_reciprocal_rank(results) == 0.0


def test_threshold_tradeoff_balances_recall_and_rejection(results) -> None:
    strict, loose = threshold_tradeoff(results, [0.4, 1.0])

    assert strict.threshold == 0.4
    assert strict.answerable_kept == pytest.approx(1 / 3)
    assert strict.unanswerable_rejected == pytest.approx(1.0)
    assert loose.answerable_kept == pytest.approx(2 / 3)
    assert loose.unanswerable_rejected == 0.0


def test_evaluate_questions_finds_rank_of_relevant_chunk(tmp_path) -> None:
    client = get_chroma_client(tmp_path / "chroma")
    collection = get_or_create_collection(client, "eval_notes")
    upsert_embedded_chunks(
        collection=collection,
        embedded_chunks=[
            make_embedded_chunk(
                "recursion_chunk",
                "Recursion needs a base case.",
                [1.0, 0.0],
                file_name="notes.pdf",
                page_number=1,
            ),
            make_embedded_chunk(
                "sql_chunk",
                "SQL queries a relational database.",
                [0.0, 1.0],
                file_name="notes.pdf",
                page_number=2,
            ),
        ],
    )
    questions = [
        EvalQuestion("What is a recursive base case?", "notes.pdf", (1,)),
        EvalQuestion("How do I query a database?", "notes.pdf", (1,)),
    ]

    results = evaluate_questions(
        questions=questions,
        collection=collection,
        embedding_model=KeywordFakeEmbeddingModel(),
        top_k=2,
    )

    assert results[0].rank == 1
    assert results[0].relevant_distance == pytest.approx(0.0, abs=1e-6)
    assert results[1].rank == 2
