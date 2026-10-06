import pytest

from src.models import EmbeddedChunk, TextChunk
from src.prompts import ANSWER_NOT_FOUND
from src.rag_pipeline import (
    EmptyQuestionError,
    answer_is_not_found,
    answer_question,
    normalise_answer,
)
from src.vector_store import (
    get_chroma_client,
    get_or_create_collection,
    upsert_embedded_chunks,
)


class FakeEmbeddingModel:
    """
    Fake embedding model for tests.

    It maps recursion-related text close to [1, 0] and database-related text
    close to [0, 1].
    """

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> list[list[float]]:
        embeddings: list[list[float]] = []

        for sentence in sentences:
            lowered = sentence.lower()

            if "database" in lowered or "sql" in lowered:
                embeddings.append([0.0, 1.0])
            else:
                embeddings.append([1.0, 0.0])

        return embeddings


class FakeChatModel:
    """
    Fake chat model for tests.
    """

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.was_called = False
        self.last_messages: list[dict[str, str]] = []

    def generate_answer(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
    ) -> str:
        self.was_called = True
        self.last_messages = messages
        return self.answer


def make_embedded_chunk(
    chunk_id: str,
    text: str,
    embedding: list[float],
    file_name: str = "lecture.pdf",
    page_number: int = 1,
) -> EmbeddedChunk:
    """
    Create a test embedded chunk.
    """

    return EmbeddedChunk(
        chunk=TextChunk(
            chunk_id=chunk_id,
            file_name=file_name,
            page_number=page_number,
            text=text,
        ),
        embedding=embedding,
    )


def test_normalise_answer_strips_whitespace() -> None:
    assert normalise_answer("  Test answer. \n") == "Test answer."


def test_answer_is_not_found_detects_required_fallback() -> None:
    assert answer_is_not_found(ANSWER_NOT_FOUND)
    assert answer_is_not_found(f"{ANSWER_NOT_FOUND}\n")


def test_answer_question_raises_for_empty_question(tmp_path) -> None:
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "rag_test_notes")

    with pytest.raises(EmptyQuestionError):
        answer_question(
            question="   ",
            collection=collection,
            embedding_model=FakeEmbeddingModel(),
            chat_model=FakeChatModel("Answer"),
            chat_model_name="fake-model",
            top_k=3,
        )


def test_answer_question_returns_fallback_when_collection_empty(tmp_path) -> None:
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "rag_test_notes")

    chat_model = FakeChatModel("This should not be called.")

    answer = answer_question(
        question="What is recursion?",
        collection=collection,
        embedding_model=FakeEmbeddingModel(),
        chat_model=chat_model,
        chat_model_name="fake-model",
        top_k=3,
    )

    assert answer.answer == ANSWER_NOT_FOUND
    assert answer.sources == []
    assert chat_model.was_called is False


def test_answer_question_retrieves_context_and_calls_chat_model(tmp_path) -> None:
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "rag_test_notes")

    chunks = [
        make_embedded_chunk(
            chunk_id="recursion_chunk",
            text="Recursion is when a function calls itself.",
            embedding=[1.0, 0.0],
            file_name="algorithms.pdf",
            page_number=4,
        ),
        make_embedded_chunk(
            chunk_id="database_chunk",
            text="A database stores structured information.",
            embedding=[0.0, 1.0],
            file_name="databases.pdf",
            page_number=2,
        ),
    ]

    upsert_embedded_chunks(collection, chunks)

    chat_model = FakeChatModel(
        "Recursion is when a function calls itself. [Source 1]"
    )

    answer = answer_question(
        question="What is recursion?",
        collection=collection,
        embedding_model=FakeEmbeddingModel(),
        chat_model=chat_model,
        chat_model_name="fake-model",
        top_k=1,
    )

    assert chat_model.was_called is True
    assert "function calls itself" in answer.answer
    assert len(answer.sources) == 1
    assert answer.sources[0].chunk_id == "recursion_chunk"
    assert answer.sources[0].file_name == "algorithms.pdf"
    assert answer.sources[0].page_number == 4

    user_message = chat_model.last_messages[1]["content"]

    assert "What is recursion?" in user_message
    assert "Recursion is when a function calls itself." in user_message
    assert "algorithms.pdf" in user_message
    assert "Page: 4" in user_message


def test_answer_question_returns_fallback_when_chat_model_says_not_found(
    tmp_path,
) -> None:
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "rag_test_notes")

    chunk = make_embedded_chunk(
        chunk_id="recursion_chunk",
        text="Recursion is when a function calls itself.",
        embedding=[1.0, 0.0],
    )

    upsert_embedded_chunks(collection, [chunk])

    chat_model = FakeChatModel(ANSWER_NOT_FOUND)

    answer = answer_question(
        question="What is quantum mechanics?",
        collection=collection,
        embedding_model=FakeEmbeddingModel(),
        chat_model=chat_model,
        chat_model_name="fake-model",
        top_k=1,
    )

    assert answer.answer == ANSWER_NOT_FOUND
    assert answer.sources == []
