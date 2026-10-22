import json

import pytest

from src.answer_evaluation import (
    JUDGE_SYSTEM_PROMPT,
    AnswerEvalResult,
    AnswerJudgement,
    JudgementParseError,
    PacedChatModel,
    answer_rate,
    build_judge_messages,
    citation_precision,
    cited_expected_page_rate,
    evaluate_answer,
    mean_faithfulness,
    mean_relevance,
    parse_judgement,
    refusal_rate,
)
from src.evaluation import EvalQuestion
from src.models import RagAnswer, RetrievedChunk
from src.prompts import ANSWER_NOT_FOUND
from src.rag_pipeline import ChatModelError
from src.vector_store import get_chroma_client, get_or_create_collection, upsert_embedded_chunks
from tests.helpers import FixedAnswerChatModel, KeywordFakeEmbeddingModel, make_embedded_chunk

SUPPORTED_JUDGEMENT = json.dumps(
    {
        "faithfulness": "supported",
        "relevance": "answers",
        "unsupported_claims": [],
        "explanation": "Everything is in the context.",
    }
)


def make_chunk(page_number: int, chunk_id: str = "", file_name: str = "notes.pdf"):
    return RetrievedChunk(
        chunk_id=chunk_id or f"chunk_p{page_number}",
        file_name=file_name,
        page_number=page_number,
        text=f"Text from page {page_number}.",
        distance=0.2,
    )


def make_result(
    pages: tuple[int, ...],
    answer: str,
    sources: list[RetrievedChunk] | None = None,
    judgement: AnswerJudgement | None = None,
    retrieved_count: int | None = None,
    error: str = "",
) -> AnswerEvalResult:
    sources = sources if sources is not None else []

    return AnswerEvalResult(
        question=EvalQuestion(question="Q?", file_name="notes.pdf", expected_pages=pages),
        answer=RagAnswer(answer=answer, sources=sources),
        retrieved_count=len(sources) if retrieved_count is None else retrieved_count,
        judgement=judgement,
        error=error,
    )


def test_parse_judgement_reads_plain_json() -> None:
    judgement = parse_judgement(SUPPORTED_JUDGEMENT)

    assert judgement == AnswerJudgement(
        faithfulness="supported",
        relevance="answers",
        unsupported_claims=(),
        explanation="Everything is in the context.",
    )
    assert judgement.faithfulness_score == 1.0
    assert judgement.relevance_score == 1.0


def test_parse_judgement_accepts_code_fences_and_label_variants() -> None:
    reply = (
        "Here is my grading:\n```json\n"
        '{"faithfulness": "Partially Supported", "relevance": "partially-answers", '
        '"unsupported_claims": "Merge sort is stable."}\n```'
    )

    judgement = parse_judgement(reply)

    assert judgement.faithfulness == "partially_supported"
    assert judgement.relevance == "partially_answers"
    assert judgement.unsupported_claims == ("Merge sort is stable.",)
    assert judgement.faithfulness_score == 0.5


@pytest.mark.parametrize(
    "reply",
    [
        "I think the answer is fine.",
        "{not json}",
        '["supported", "answers"]',
        '{"faithfulness": "mostly", "relevance": "answers"}',
        '{"faithfulness": "supported", "relevance": "sort of"}',
    ],
)
def test_parse_judgement_rejects_invalid_replies(reply: str) -> None:
    with pytest.raises(JudgementParseError):
        parse_judgement(reply)


def test_build_judge_messages_includes_question_context_and_answer() -> None:
    messages = build_judge_messages(
        question="What is a heap?",
        answer="A heap is a tree. [Source 1]",
        sources=[make_chunk(5)],
    )

    assert messages[0] == {"role": "system", "content": JUDGE_SYSTEM_PROMPT}
    user_prompt = messages[1]["content"]
    assert "What is a heap?" in user_prompt
    assert "[Source 1]" in user_prompt
    assert "Text from page 5." in user_prompt
    assert "A heap is a tree. [Source 1]" in user_prompt


def test_result_maps_citations_to_pages() -> None:
    result = make_result(
        pages=(2,),
        answer="Arrays allow random access. [Source 1][Source 2] Also [Source 9].",
        sources=[make_chunk(2), make_chunk(3), make_chunk(2, chunk_id="other")],
    )

    assert result.cited_pages == [2, 3]
    assert result.relevant_citation_count == 1
    assert result.refused is False


def test_answer_and_refusal_rates() -> None:
    results = [
        make_result(pages=(1,), answer="Yes. [Source 1]", sources=[make_chunk(1)]),
        make_result(pages=(1,), answer=ANSWER_NOT_FOUND, retrieved_count=2),
        make_result(pages=(), answer=ANSWER_NOT_FOUND, retrieved_count=0),
        make_result(pages=(), answer="Made up. [Source 1]", sources=[make_chunk(4)]),
        make_result(pages=(), answer=ANSWER_NOT_FOUND, error="Answer failed: rate limit"),
    ]

    assert answer_rate(results) == 0.5
    # The errored question is left out instead of counting as a refusal.
    assert refusal_rate(results) == 0.5
    assert results[2].rejected_before_model is True
    assert results[1].rejected_before_model is False


def test_citation_metrics_only_count_answered_answerable_questions() -> None:
    results = [
        make_result(
            pages=(1,),
            answer="A. [Source 1][Source 2]",
            sources=[make_chunk(1), make_chunk(6)],
        ),
        make_result(pages=(3,), answer="B. [Source 1]", sources=[make_chunk(4)]),
        make_result(pages=(3,), answer=ANSWER_NOT_FOUND, retrieved_count=1),
        make_result(pages=(), answer="C. [Source 1]", sources=[make_chunk(7)]),
    ]

    assert cited_expected_page_rate(results) == 0.5
    assert citation_precision(results) == pytest.approx(1 / 3)


def test_judge_score_means_ignore_unjudged_results() -> None:
    results = [
        make_result(
            pages=(1,),
            answer="A. [Source 1]",
            judgement=AnswerJudgement(faithfulness="supported", relevance="answers"),
        ),
        make_result(
            pages=(1,),
            answer="B. [Source 1]",
            judgement=AnswerJudgement(
                faithfulness="partially_supported",
                relevance="does_not_answer",
            ),
        ),
        make_result(pages=(1,), answer=ANSWER_NOT_FOUND),
    ]

    assert mean_faithfulness(results) == 0.75
    assert mean_relevance(results) == 0.5


def test_metrics_return_none_without_data() -> None:
    assert answer_rate([]) is None
    assert refusal_rate([]) is None
    assert citation_precision([]) is None
    assert mean_faithfulness([]) is None


def store_notes(tmp_path):
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "answer_eval_notes")
    upsert_embedded_chunks(
        collection,
        [
            make_embedded_chunk(
                chunk_id="recursion_chunk",
                text="Recursion needs a base case.",
                embedding=[1.0, 0.0],
                file_name="notes.pdf",
                page_number=8,
            ),
            make_embedded_chunk(
                chunk_id="database_chunk",
                text="SQL queries relational databases.",
                embedding=[0.0, 1.0],
                file_name="notes.pdf",
                page_number=3,
            ),
        ],
    )
    return collection


def run_evaluation(collection, question: EvalQuestion, chat_model, judge_model, max_distance=0.5):
    return evaluate_answer(
        question=question,
        collection=collection,
        embedding_model=KeywordFakeEmbeddingModel(),
        chat_model=chat_model,
        chat_model_name="fake-model",
        judge_model=judge_model,
        judge_model_name="fake-judge",
        top_k=1,
        max_retrieval_distance=max_distance,
    )


def test_evaluate_answer_answers_and_grades(tmp_path) -> None:
    collection = store_notes(tmp_path)
    chat_model = FixedAnswerChatModel("Recursion needs a base case. [Source 1]")
    judge_model = FixedAnswerChatModel(SUPPORTED_JUDGEMENT)
    question = EvalQuestion("What does recursion need?", "notes.pdf", (8,))

    result = run_evaluation(collection, question, chat_model, judge_model)

    assert result.error == ""
    assert result.refused is False
    assert result.cited_pages == [8]
    assert result.judgement.faithfulness == "supported"
    judge_prompt = judge_model.last_messages[1]["content"]
    assert "Recursion needs a base case." in judge_prompt
    assert "What does recursion need?" in judge_prompt


def test_evaluate_answer_skips_model_and_judge_when_retrieval_finds_nothing(tmp_path) -> None:
    collection = store_notes(tmp_path)
    chat_model = FixedAnswerChatModel("unused")
    judge_model = FixedAnswerChatModel(SUPPORTED_JUDGEMENT)
    question = EvalQuestion("What is the capital of Finland?", "notes.pdf", ())

    result = run_evaluation(collection, question, chat_model, judge_model, max_distance=0.2)

    assert result.refused is True
    assert result.rejected_before_model is True
    assert chat_model.was_called is False
    assert judge_model.was_called is False


def test_evaluate_answer_does_not_judge_model_refusals(tmp_path) -> None:
    collection = store_notes(tmp_path)
    judge_model = FixedAnswerChatModel(SUPPORTED_JUDGEMENT)
    question = EvalQuestion("How does recursion relate to B-trees?", "notes.pdf", ())

    result = run_evaluation(
        collection, question, FixedAnswerChatModel(ANSWER_NOT_FOUND), judge_model
    )

    assert result.refused is True
    assert result.rejected_before_model is False
    assert judge_model.was_called is False


def test_evaluate_answer_records_judge_and_answer_failures(tmp_path) -> None:
    class FailingChatModel(FixedAnswerChatModel):
        def generate_answer(self, messages, model, temperature=0.0):
            raise ChatModelError("The chat API rate limit or quota was reached.")

    collection = store_notes(tmp_path)
    question = EvalQuestion("What does recursion need?", "notes.pdf", (8,))

    bad_judge = run_evaluation(
        collection,
        question,
        FixedAnswerChatModel("Recursion needs a base case. [Source 1]"),
        FixedAnswerChatModel("Looks good to me!"),
    )
    failed_answer = run_evaluation(
        collection,
        question,
        FailingChatModel("unused"),
        FixedAnswerChatModel(SUPPORTED_JUDGEMENT),
    )

    assert bad_judge.error.startswith("Judge failed")
    assert bad_judge.judgement is None
    assert failed_answer.error.startswith("Answer failed")
    assert "rate limit" in failed_answer.error


def test_paced_chat_model_spaces_out_requests() -> None:
    now = [100.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    inner = FixedAnswerChatModel("Answer")
    paced = PacedChatModel(inner, requests_per_minute=30, clock=lambda: now[0], sleep=sleep)
    messages = [{"role": "user", "content": "Hi"}]

    assert paced.generate_answer(messages, model="fake") == "Answer"
    now[0] += 0.5
    assert "".join(paced.stream_answer(messages, model="fake")) == "Answer"
    now[0] += 5.0
    paced.generate_answer(messages, model="fake")

    assert sleeps == [pytest.approx(1.5)]
    assert len(inner.calls) == 3


def test_paced_chat_model_rejects_invalid_rate() -> None:
    with pytest.raises(ValueError):
        PacedChatModel(FixedAnswerChatModel("x"), requests_per_minute=0)
