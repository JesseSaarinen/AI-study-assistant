import csv
import io
import json
import random
from dataclasses import dataclass
from typing import Any

from src.embeddings import EmbeddingModel
from src.models import RetrievedChunk
from src.prompts import build_context_block
from src.rag_pipeline import ChatModel, ChatModelError, retrieve_relevant_chunks_for_question
from src.vector_store import VectorStoreError, get_collection_preview

MIN_QUESTIONS = 1
MAX_QUESTIONS = 10

# Upper bound on how many stored chunks are considered when no topic is given.
QUIZ_CANDIDATE_LIMIT = 2000

# A short topic quiz still gets a few passages, so the model has some choice.
MIN_TOPIC_PASSAGES = 3

QUIZ_SYSTEM_PROMPT = """
You write multiple-choice practice questions for a university student, using
ONLY the provided passages from their lecture notes.

Rules:
1. Every question must be answerable from one passage. Do not use outside knowledge.
2. Test understanding, such as why, when, or how two ideas compare, rather than
   trivia such as lecture titles.
3. Give exactly 4 options. Exactly one is correct according to the notes, and the
   wrong options must be plausible but clearly wrong according to the notes.
4. Write in the language of the passages.
5. The explanation says in one or two sentences why the correct option is right.
6. "source" is the number N of the [Source N] passage that supports the answer.

Reply with only a JSON array and no Markdown, in this form:
[
  {
    "question": "...",
    "options": ["...", "...", "...", "..."],
    "correct_option": "the correct option, copied exactly from options",
    "explanation": "...",
    "source": 1
  }
]
""".strip()


class QuizGenerationError(Exception):
    """
    Raised when a quiz cannot be generated.
    """


@dataclass(frozen=True)
class QuizQuestion:
    """
    One multiple-choice question with the notes page that supports its answer.
    """

    question: str
    options: tuple[str, ...]
    correct_index: int
    explanation: str
    file_name: str
    page_number: int

    @property
    def correct_option(self) -> str:
        return self.options[self.correct_index]


def select_quiz_chunks(
    collection: Any,
    embedding_model: EmbeddingModel,
    question_count: int,
    topic: str = "",
    max_retrieval_distance: float | None = None,
    rng: random.Random | None = None,
) -> list[RetrievedChunk]:
    """
    Pick the passages a quiz is written from.

    With a topic, the chunks closest to the topic are used, the same way a
    question is answered. Without one, chunks are drawn from as many different
    pages as possible, so a quiz covers the whole set of notes.
    """

    if topic.strip():
        return retrieve_relevant_chunks_for_question(
            question=topic,
            collection=collection,
            embedding_model=embedding_model,
            top_k=max(question_count, MIN_TOPIC_PASSAGES),
            max_retrieval_distance=max_retrieval_distance,
        )

    rng = rng or random.Random()

    try:
        chunks = get_collection_preview(collection, limit=QUIZ_CANDIDATE_LIMIT)
    except VectorStoreError as exc:
        raise QuizGenerationError("Could not read the stored notes.") from exc

    chunks_by_page: dict[tuple[str, int], list[RetrievedChunk]] = {}

    for chunk in chunks:
        chunks_by_page.setdefault((chunk.file_name, chunk.page_number), []).append(chunk)

    page_groups = list(chunks_by_page.values())
    rng.shuffle(page_groups)

    for group in page_groups:
        rng.shuffle(group)

    # Take one chunk from each page in turn until there are enough.
    selected: list[RetrievedChunk] = []

    while len(selected) < question_count and any(page_groups):
        for group in page_groups:
            if group and len(selected) < question_count:
                selected.append(group.pop())

    return selected


def build_quiz_messages(
    chunks: list[RetrievedChunk],
    question_count: int,
) -> list[dict[str, str]]:
    """
    Build chat messages that ask for multiple-choice questions about the chunks.
    """

    user_prompt = f"""
Write {question_count} multiple-choice question(s), using different passages where possible.

Passages from the lecture notes:
{build_context_block(chunks)}
""".strip()

    return [
        {
            "role": "system",
            "content": QUIZ_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def _extract_json_items(raw_reply: str) -> list[Any]:
    start = raw_reply.find("[")
    end = raw_reply.rfind("]")

    if start == -1 or end <= start:
        raise QuizGenerationError("The model did not return a list of questions.")

    try:
        data = json.loads(raw_reply[start : end + 1])
    except json.JSONDecodeError as exc:
        raise QuizGenerationError("The model returned questions that are not valid JSON.") from exc

    if not isinstance(data, list):
        raise QuizGenerationError("The model did not return a list of questions.")

    return data


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


def _parse_item(
    item: Any,
    chunks: list[RetrievedChunk],
    rng: random.Random,
) -> QuizQuestion | None:
    """
    Turn one JSON item into a QuizQuestion, or return None if it is unusable.
    """

    if not isinstance(item, dict):
        return None

    question = str(item.get("question", "")).strip()
    raw_options = item.get("options")
    correct = _normalise(str(item.get("correct_option", "")))

    if not question or not isinstance(raw_options, list):
        return None

    options = [str(option).strip() for option in raw_options if str(option).strip()]

    if not 2 <= len(options) <= 6 or len({_normalise(option) for option in options}) != len(
        options
    ):
        return None

    matches = [index for index, option in enumerate(options) if _normalise(option) == correct]

    if len(matches) != 1:
        return None

    try:
        source_number = int(item.get("source", 0))
    except (TypeError, ValueError):
        return None

    if not 1 <= source_number <= len(chunks):
        return None

    source = chunks[source_number - 1]

    # Models tend to put the correct answer in the same position, so the
    # options are shuffled here rather than trusting the model's order.
    correct_option = options[matches[0]]
    rng.shuffle(options)

    return QuizQuestion(
        question=question,
        options=tuple(options),
        correct_index=options.index(correct_option),
        explanation=str(item.get("explanation", "")).strip(),
        file_name=source.file_name,
        page_number=source.page_number,
    )


def parse_quiz(
    raw_reply: str,
    chunks: list[RetrievedChunk],
    rng: random.Random | None = None,
) -> list[QuizQuestion]:
    """
    Parse the model's JSON reply into quiz questions.

    Items with a missing answer, duplicate options or an unknown source are
    skipped instead of failing the whole quiz.
    """

    rng = rng or random.Random()
    questions = [
        question
        for question in (_parse_item(item, chunks, rng) for item in _extract_json_items(raw_reply))
        if question is not None
    ]

    if not questions:
        raise QuizGenerationError("The model's reply did not contain any usable questions.")

    return questions


def generate_quiz(
    collection: Any,
    embedding_model: EmbeddingModel,
    chat_model: ChatModel,
    chat_model_name: str,
    question_count: int = 5,
    topic: str = "",
    max_retrieval_distance: float | None = None,
    rng: random.Random | None = None,
) -> list[QuizQuestion]:
    """
    Generate multiple-choice questions from the stored notes.
    """

    if not MIN_QUESTIONS <= question_count <= MAX_QUESTIONS:
        raise QuizGenerationError(f"Choose between {MIN_QUESTIONS} and {MAX_QUESTIONS} questions.")

    rng = rng or random.Random()

    chunks = select_quiz_chunks(
        collection=collection,
        embedding_model=embedding_model,
        question_count=question_count,
        topic=topic,
        max_retrieval_distance=max_retrieval_distance,
        rng=rng,
    )

    if not chunks:
        if topic.strip():
            raise QuizGenerationError(
                "No part of the notes is close enough to this topic. Try another topic."
            )

        raise QuizGenerationError("Process some notes before generating a quiz.")

    try:
        reply = chat_model.generate_answer(
            messages=build_quiz_messages(chunks, question_count),
            model=chat_model_name,
            temperature=0.3,
        )
    except ChatModelError:
        raise
    except Exception as exc:
        raise ChatModelError("Chat model failed to generate a quiz.") from exc

    return parse_quiz(reply, chunks, rng)[:question_count]


def quiz_to_flashcards_csv(questions: list[QuizQuestion]) -> str:
    """
    Export quiz questions as front and back flashcards in CSV.

    The two-column format imports into Anki and most other flashcard apps.
    """

    output = io.StringIO()
    writer = csv.writer(output)

    for question in questions:
        back = question.correct_option

        if question.explanation:
            back = f"{back}\n\n{question.explanation}"

        back = f"{back}\n\n({question.file_name}, page {question.page_number})"
        writer.writerow([question.question, back])

    return output.getvalue()
