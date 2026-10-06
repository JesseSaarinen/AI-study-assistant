from src.models import EmbeddedChunk, TextChunk
from src.prompts import ANSWER_NOT_FOUND
from src.rag_pipeline import answer_question
from src.vector_store import (
    get_chroma_client,
    get_or_create_collection,
    upsert_embedded_chunks,
)


class FakeEmbeddingModel:
    """
    Fake embedding model for tests.

    Recursion-like text maps to [1, 0].
    Database-like text maps to [0, 1].
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
    Fake chat model that records whether it was called.
    """

    def __init__(self) -> None:
        self.was_called = False

    def generate_answer(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
    ) -> str:
        self.was_called = True
        return "This should not be used."


def make_embedded_chunk(
    chunk_id: str,
    text: str,
    embedding: list[float],
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=TextChunk(
            chunk_id=chunk_id,
            file_name="lecture.pdf",
            page_number=1,
            text=text,
        ),
        embedding=embedding,
    )


def test_answer_question_returns_fallback_when_retrieval_distance_too_high(
    tmp_path,
) -> None:
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "rag_robust_notes")

    chunk = make_embedded_chunk(
        chunk_id="recursion_chunk",
        text="Recursion is when a function calls itself.",
        embedding=[1.0, 0.0],
    )

    upsert_embedded_chunks(collection, [chunk])

    chat_model = FakeChatModel()

    answer = answer_question(
        question="What is SQL?",
        collection=collection,
        embedding_model=FakeEmbeddingModel(),
        chat_model=chat_model,
        chat_model_name="fake-model",
        top_k=1,
        max_retrieval_distance=0.5,
    )

    assert answer.answer == ANSWER_NOT_FOUND
    assert answer.sources == []
    assert answer.citations == []
    assert chat_model.was_called is False
