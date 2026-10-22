import csv
import io
import json
import random

import pytest

from src.models import RetrievedChunk
from src.quiz import (
    QUIZ_SYSTEM_PROMPT,
    QuizGenerationError,
    QuizQuestion,
    build_quiz_messages,
    generate_quiz,
    parse_quiz,
    quiz_to_flashcards_csv,
    select_quiz_chunks,
)
from src.vector_store import get_chroma_client, get_or_create_collection, upsert_embedded_chunks
from tests.helpers import FixedAnswerChatModel, KeywordFakeEmbeddingModel, make_embedded_chunk


def make_chunk(page_number: int, text: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk_p{page_number}",
        file_name="notes.pdf",
        page_number=page_number,
        text=text or f"Text from page {page_number}.",
    )


def quiz_item(
    question: str = "Why does a dynamic array double its capacity?",
    options: list[str] | None = None,
    correct: str = "To keep appends O(1) amortised",
    source: int = 1,
) -> dict:
    return {
        "question": question,
        "options": options
        or [
            "To keep appends O(1) amortised",
            "To save memory",
            "To keep elements sorted",
            "To allow random access",
        ],
        "correct_option": correct,
        "explanation": "Doubling makes resizes rare.",
        "source": source,
    }


def store_chunks(tmp_path, pages_and_texts: list[tuple[int, str, list[float]]]):
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "quiz_notes")
    upsert_embedded_chunks(
        collection,
        [
            make_embedded_chunk(
                chunk_id=f"chunk_{index}",
                text=text,
                embedding=embedding,
                file_name="notes.pdf",
                page_number=page,
            )
            for index, (page, text, embedding) in enumerate(pages_and_texts)
        ],
    )
    return collection


def test_select_quiz_chunks_spreads_questions_across_pages(tmp_path) -> None:
    collection = store_chunks(
        tmp_path,
        [
            (1, "Page one, part one.", [0.5, 0.5]),
            (1, "Page one, part two.", [0.5, 0.5]),
            (1, "Page one, part three.", [0.5, 0.5]),
            (2, "Page two.", [0.5, 0.5]),
            (3, "Page three.", [0.5, 0.5]),
        ],
    )

    chunks = select_quiz_chunks(
        collection,
        KeywordFakeEmbeddingModel(),
        question_count=3,
        rng=random.Random(7),
    )

    assert sorted(chunk.page_number for chunk in chunks) == [1, 2, 3]

    all_chunks = select_quiz_chunks(
        collection,
        KeywordFakeEmbeddingModel(),
        question_count=10,
        rng=random.Random(7),
    )

    assert len(all_chunks) == 5


def test_select_quiz_chunks_uses_closest_chunks_for_a_topic(tmp_path) -> None:
    collection = store_chunks(
        tmp_path,
        [
            (1, "Recursion needs a base case.", [1.0, 0.0]),
            (2, "SQL queries relational databases.", [0.0, 1.0]),
        ],
    )

    chunks = select_quiz_chunks(
        collection,
        KeywordFakeEmbeddingModel(),
        question_count=1,
        topic="recursion",
        max_retrieval_distance=0.5,
    )

    assert [chunk.page_number for chunk in chunks] == [1]


def test_build_quiz_messages_lists_sources_and_count() -> None:
    messages = build_quiz_messages([make_chunk(2), make_chunk(5)], question_count=3)

    assert messages[0] == {"role": "system", "content": QUIZ_SYSTEM_PROMPT}
    assert "Write 3 multiple-choice question(s)" in messages[1]["content"]
    assert "[Source 2]" in messages[1]["content"]
    assert "Page: 5" in messages[1]["content"]


def test_parse_quiz_maps_source_and_keeps_correct_answer_after_shuffle() -> None:
    reply = "```json\n" + json.dumps([quiz_item(source=2)]) + "\n```"

    for seed in range(5):
        questions = parse_quiz(reply, [make_chunk(1), make_chunk(4)], random.Random(seed))

        assert len(questions) == 1
        assert questions[0].correct_option == "To keep appends O(1) amortised"
        assert questions[0].page_number == 4
        assert questions[0].explanation == "Doubling makes resizes rare."
        assert sorted(questions[0].options) == sorted(quiz_item()["options"])


def test_parse_quiz_shuffles_option_order() -> None:
    reply = json.dumps([quiz_item()])
    positions = {
        parse_quiz(reply, [make_chunk(1)], random.Random(seed))[0].correct_index
        for seed in range(20)
    }

    assert len(positions) > 1


def test_parse_quiz_skips_unusable_items() -> None:
    reply = json.dumps(
        [
            quiz_item(question="Good question?"),
            quiz_item(question="No matching answer?", correct="Something else"),
            quiz_item(question="Duplicate options?", options=["A", "a", "B", "C"], correct="B"),
            quiz_item(question="Unknown source?", source=7),
            {"question": "No options?"},
            "not an object",
        ]
    )

    questions = parse_quiz(reply, [make_chunk(1)], random.Random(0))

    assert [question.question for question in questions] == ["Good question?"]


@pytest.mark.parametrize(
    "reply",
    [
        "Sorry, I cannot write a quiz.",
        "[not json]",
        json.dumps([quiz_item(source=9)]),
    ],
)
def test_parse_quiz_raises_without_usable_questions(reply: str) -> None:
    with pytest.raises(QuizGenerationError):
        parse_quiz(reply, [make_chunk(1)])


def test_generate_quiz_writes_questions_from_stored_notes(tmp_path) -> None:
    collection = store_chunks(
        tmp_path,
        [
            (1, "Dynamic arrays double their capacity.", [0.5, 0.5]),
            (2, "Recursion needs a base case.", [1.0, 0.0]),
        ],
    )
    chat_model = FixedAnswerChatModel(
        "unused",
        quiz=json.dumps([quiz_item(question=f"Question {number}?") for number in range(4)]),
    )

    questions = generate_quiz(
        collection=collection,
        embedding_model=KeywordFakeEmbeddingModel(),
        chat_model=chat_model,
        chat_model_name="fake-model",
        question_count=2,
        rng=random.Random(1),
    )

    assert [question.question for question in questions] == ["Question 0?", "Question 1?"]
    assert "Dynamic arrays double their capacity." in chat_model.last_messages[1]["content"]


def test_generate_quiz_explains_missing_notes_and_topics(tmp_path) -> None:
    empty = get_or_create_collection(get_chroma_client(tmp_path), "empty_quiz_notes")
    collection = store_chunks(tmp_path, [(1, "Recursion needs a base case.", [1.0, 0.0])])
    chat_model = FixedAnswerChatModel("unused")

    with pytest.raises(QuizGenerationError, match="Process some notes"):
        generate_quiz(empty, KeywordFakeEmbeddingModel(), chat_model, "fake-model")

    with pytest.raises(QuizGenerationError, match="topic"):
        generate_quiz(
            collection,
            KeywordFakeEmbeddingModel(),
            chat_model,
            "fake-model",
            topic="relational databases",
            max_retrieval_distance=0.5,
        )

    with pytest.raises(QuizGenerationError, match="between"):
        generate_quiz(collection, KeywordFakeEmbeddingModel(), chat_model, "fake", 0)

    assert chat_model.was_called is False


def test_quiz_to_flashcards_csv_has_question_and_answer_columns() -> None:
    question = QuizQuestion(
        question='What does "LIFO" mean?',
        options=("Last in, first out", "First in, first out"),
        correct_index=0,
        explanation="A stack removes the newest item first.",
        file_name="notes.pdf",
        page_number=3,
    )

    rows = list(csv.reader(io.StringIO(quiz_to_flashcards_csv([question]))))

    assert len(rows) == 1
    front, back = rows[0]
    assert front == 'What does "LIFO" mean?'
    assert back.startswith("Last in, first out")
    assert "A stack removes the newest item first." in back
    assert back.endswith("(notes.pdf, page 3)")
