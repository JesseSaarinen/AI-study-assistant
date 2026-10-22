import json
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from src.citations import extract_source_numbers
from src.embeddings import EmbeddingModel
from src.evaluation import EvalQuestion
from src.models import RagAnswer, RetrievedChunk
from src.prompts import ANSWER_NOT_FOUND, build_context_block
from src.rag_pipeline import ChatModel, ChatModelError, finalise_answer, prepare_answer

FAITHFULNESS_SCORES = {
    "supported": 1.0,
    "partially_supported": 0.5,
    "unsupported": 0.0,
}

RELEVANCE_SCORES = {
    "answers": 1.0,
    "partially_answers": 0.5,
    "does_not_answer": 0.0,
}


JUDGE_SYSTEM_PROMPT = """
You grade answers from a study assistant that must answer only from lecture notes.

You get a question, the context passages the assistant was given, and the
assistant's answer. Grade two things.

faithfulness: is every factual claim in the answer supported by the context?
- "supported": every claim is stated in the context or follows directly from it.
- "partially_supported": the main point is supported, but some details are not in the context.
- "unsupported": the main point is not in the context, or the answer contradicts the context.

relevance: does the answer address the question?
- "answers": it answers the question directly and completely.
- "partially_answers": it answers only part of the question, or stays vague.
- "does_not_answer": it does not answer the question.

Ignore citation markers such as [Source 1]. Do not use your own knowledge to
decide whether a claim is true: a claim that is correct but not in the context
counts as unsupported.

Reply with only a JSON object and no Markdown, in this form:
{
  "faithfulness": "supported | partially_supported | unsupported",
  "relevance": "answers | partially_answers | does_not_answer",
  "unsupported_claims": ["each claim that is not in the context"],
  "explanation": "one sentence"
}
""".strip()


class JudgementParseError(ValueError):
    """
    Raised when the judge model's reply is not a valid judgement.
    """


@dataclass(frozen=True)
class AnswerJudgement:
    """
    The judge model's verdict on one answer.
    """

    faithfulness: str
    relevance: str
    unsupported_claims: tuple[str, ...] = ()
    explanation: str = ""

    @property
    def faithfulness_score(self) -> float:
        return FAITHFULNESS_SCORES[self.faithfulness]

    @property
    def relevance_score(self) -> float:
        return RELEVANCE_SCORES[self.relevance]


@dataclass(frozen=True)
class AnswerEvalResult:
    """
    The app's answer to one test question, and how it was graded.

    retrieved_count is the number of chunks that passed the distance threshold.
    Zero means the app said the answer is not in the notes without calling the
    chat model.
    """

    question: EvalQuestion
    answer: RagAnswer
    retrieved_count: int
    judgement: AnswerJudgement | None = None
    error: str = ""

    @property
    def refused(self) -> bool:
        return self.answer.answer == ANSWER_NOT_FOUND

    @property
    def rejected_before_model(self) -> bool:
        return self.retrieved_count == 0

    @property
    def cited_sources(self) -> list[RetrievedChunk]:
        sources = self.answer.sources

        return [
            sources[number - 1]
            for number in extract_source_numbers(self.answer.answer)
            if 1 <= number <= len(sources)
        ]

    @property
    def cited_pages(self) -> list[int]:
        return sorted({source.page_number for source in self.cited_sources})

    @property
    def relevant_citation_count(self) -> int:
        return sum(1 for source in self.cited_sources if self._is_expected(source))

    def _is_expected(self, source: RetrievedChunk) -> bool:
        return (
            source.file_name == self.question.file_name
            and source.page_number in self.question.expected_pages
        )


def build_judge_messages(
    question: str,
    answer: str,
    sources: list[RetrievedChunk],
) -> list[dict[str, str]]:
    """
    Build chat messages that ask the judge model to grade one answer.
    """

    user_prompt = f"""
Question:
{question}

Context given to the assistant:
{build_context_block(sources)}

Assistant's answer:
{answer}
""".strip()

    return [
        {
            "role": "system",
            "content": JUDGE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def _normalise_label(value: Any) -> str:
    return re.sub(r"[\s-]+", "_", str(value).strip().lower())


def parse_judgement(raw_reply: str) -> AnswerJudgement:
    """
    Parse the judge model's JSON reply.

    Models sometimes wrap JSON in a Markdown code block or add a sentence
    around it, so the first JSON object in the reply is used.
    """

    start = raw_reply.find("{")
    end = raw_reply.rfind("}")

    if start == -1 or end <= start:
        raise JudgementParseError("The judge reply contains no JSON object.")

    try:
        data = json.loads(raw_reply[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgementParseError("The judge reply is not valid JSON.") from exc

    if not isinstance(data, dict):
        raise JudgementParseError("The judge reply is not a JSON object.")

    faithfulness = _normalise_label(data.get("faithfulness", ""))
    relevance = _normalise_label(data.get("relevance", ""))

    if faithfulness not in FAITHFULNESS_SCORES:
        raise JudgementParseError(f"Unknown faithfulness label: {faithfulness!r}")

    if relevance not in RELEVANCE_SCORES:
        raise JudgementParseError(f"Unknown relevance label: {relevance!r}")

    raw_claims = data.get("unsupported_claims") or []

    if isinstance(raw_claims, str):
        raw_claims = [raw_claims]

    claims = tuple(str(claim).strip() for claim in raw_claims if str(claim).strip())

    return AnswerJudgement(
        faithfulness=faithfulness,
        relevance=relevance,
        unsupported_claims=claims,
        explanation=str(data.get("explanation", "")).strip(),
    )


class PacedChatModel:
    """
    Chat model wrapper that waits between requests.

    Free API tiers allow only a few requests per minute. Spacing the requests
    out keeps a full evaluation run under that limit.
    """

    def __init__(
        self,
        chat_model: ChatModel,
        requests_per_minute: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than 0.")

        self.chat_model = chat_model
        self.min_interval = 60.0 / requests_per_minute
        self.clock = clock
        self.sleep = sleep
        self.last_request_at: float | None = None

    def _wait_for_turn(self) -> None:
        if self.last_request_at is not None:
            remaining = self.min_interval - (self.clock() - self.last_request_at)

            if remaining > 0:
                self.sleep(remaining)

        self.last_request_at = self.clock()

    def generate_answer(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
    ) -> str:
        self._wait_for_turn()
        return self.chat_model.generate_answer(messages, model, temperature)

    def stream_answer(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        self._wait_for_turn()
        return self.chat_model.stream_answer(messages, model, temperature)


def evaluate_answer(
    question: EvalQuestion,
    collection: Any,
    embedding_model: EmbeddingModel,
    chat_model: ChatModel,
    chat_model_name: str,
    judge_model: ChatModel,
    judge_model_name: str,
    top_k: int,
    max_retrieval_distance: float | None,
) -> AnswerEvalResult:
    """
    Answer one test question with the app's pipeline and grade the answer.

    Refusals are not sent to the judge: whether a refusal is right is decided
    by the question's label alone.
    """

    prepared = prepare_answer(
        question=question.question,
        collection=collection,
        embedding_model=embedding_model,
        chat_model=chat_model,
        chat_model_name=chat_model_name,
        top_k=top_k,
        max_retrieval_distance=max_retrieval_distance,
    )

    raw_answer = ""

    if prepared.needs_chat_model:
        try:
            raw_answer = chat_model.generate_answer(
                messages=prepared.messages,
                model=chat_model_name,
                temperature=0.0,
            )
        except ChatModelError as exc:
            return AnswerEvalResult(
                question=question,
                answer=finalise_answer("", prepared),
                retrieved_count=len(prepared.sources),
                error=f"Answer failed: {exc}",
            )

    answer = finalise_answer(raw_answer, prepared)
    result = AnswerEvalResult(
        question=question,
        answer=answer,
        retrieved_count=len(prepared.sources),
    )

    if result.refused:
        return result

    try:
        reply = judge_model.generate_answer(
            messages=build_judge_messages(
                question=question.question,
                answer=answer.answer,
                sources=answer.sources,
            ),
            model=judge_model_name,
            temperature=0.0,
        )
        judgement = parse_judgement(reply)
    except (ChatModelError, JudgementParseError) as exc:
        return AnswerEvalResult(
            question=question,
            answer=answer,
            retrieved_count=result.retrieved_count,
            error=f"Judge failed: {exc}",
        )

    return AnswerEvalResult(
        question=question,
        answer=answer,
        retrieved_count=result.retrieved_count,
        judgement=judgement,
    )


def _answerable(results: list[AnswerEvalResult]) -> list[AnswerEvalResult]:
    return [result for result in results if result.question.answerable and not result.error]


def _unanswerable(results: list[AnswerEvalResult]) -> list[AnswerEvalResult]:
    return [result for result in results if not result.question.answerable and not result.error]


def _share(count: float, total: int) -> float | None:
    return count / total if total else None


def answer_rate(results: list[AnswerEvalResult]) -> float | None:
    """
    Share of answerable questions that got an answer instead of a refusal.
    """

    answerable = _answerable(results)

    return _share(sum(1 for result in answerable if not result.refused), len(answerable))


def refusal_rate(results: list[AnswerEvalResult]) -> float | None:
    """
    Share of unanswerable questions where the app said the answer is not in the notes.
    """

    unanswerable = _unanswerable(results)

    return _share(sum(1 for result in unanswerable if result.refused), len(unanswerable))


def cited_expected_page_rate(results: list[AnswerEvalResult]) -> float | None:
    """
    Share of answered, answerable questions that cite at least one expected page.
    """

    answered = [result for result in _answerable(results) if not result.refused]

    return _share(
        sum(1 for result in answered if result.relevant_citation_count > 0),
        len(answered),
    )


def citation_precision(results: list[AnswerEvalResult]) -> float | None:
    """
    Share of all citations in answered, answerable questions that point to an expected page.

    Pages next to the labelled ones can support an answer too, so this is a
    lower bound on how often citations are actually right.
    """

    answered = [result for result in _answerable(results) if not result.refused]
    cited = sum(len(result.cited_sources) for result in answered)

    return _share(sum(result.relevant_citation_count for result in answered), cited)


def _judged(results: list[AnswerEvalResult]) -> list[AnswerJudgement]:
    return [result.judgement for result in results if result.judgement is not None]


def mean_faithfulness(results: list[AnswerEvalResult]) -> float | None:
    """
    Mean faithfulness score over all graded answers.
    """

    judged = _judged(results)

    return _share(sum(judgement.faithfulness_score for judgement in judged), len(judged))


def mean_relevance(results: list[AnswerEvalResult]) -> float | None:
    """
    Mean relevance score over all graded answers.
    """

    judged = _judged(results)

    return _share(sum(judgement.relevance_score for judgement in judged), len(judged))
