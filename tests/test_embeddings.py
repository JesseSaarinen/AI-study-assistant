import pytest

from src.embeddings import (
    EmbeddingGenerationError,
    embed_chunks,
    generate_embeddings,
    get_embedding_dimension,
)
from src.models import TextChunk


class FakeEmbeddingModel:
    """
    Fake model used for tests.

    It behaves like a Sentence Transformers model but returns small deterministic
    vectors instead of real embeddings.
    """

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> list[list[float]]:
        return [
            [float(len(sentence)), float(index)]
            for index, sentence in enumerate(sentences)
        ]


class BadCountEmbeddingModel:
    """
    Fake broken model that returns the wrong number of embeddings.
    """

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> list[list[float]]:
        return [[1.0, 2.0]]


def test_generate_embeddings_returns_embedding_for_each_text() -> None:
    model = FakeEmbeddingModel()
    texts = ["first chunk", "second chunk"]

    embeddings = generate_embeddings(texts=texts, model=model)

    assert len(embeddings) == 2
    assert embeddings[0] == [11.0, 0.0]
    assert embeddings[1] == [12.0, 1.0]


def test_generate_embeddings_returns_empty_list_for_no_texts() -> None:
    model = FakeEmbeddingModel()

    embeddings = generate_embeddings(texts=[], model=model)

    assert embeddings == []


def test_generate_embeddings_rejects_empty_text() -> None:
    model = FakeEmbeddingModel()

    with pytest.raises(EmbeddingGenerationError):
        generate_embeddings(texts=["valid text", "   "], model=model)


def test_embed_chunks_preserves_metadata() -> None:
    model = FakeEmbeddingModel()

    chunks = [
        TextChunk(
            chunk_id="lecture_1_p1_c1",
            file_name="lecture_1.pdf",
            page_number=1,
            text="This is chunk one.",
        ),
        TextChunk(
            chunk_id="lecture_1_p2_c1",
            file_name="lecture_1.pdf",
            page_number=2,
            text="This is chunk two.",
        ),
    ]

    embedded_chunks = embed_chunks(chunks=chunks, model=model)

    assert len(embedded_chunks) == 2

    assert embedded_chunks[0].chunk == chunks[0]
    assert embedded_chunks[0].embedding == [18.0, 0.0]

    assert embedded_chunks[1].chunk == chunks[1]
    assert embedded_chunks[1].embedding == [18.0, 1.0]


def test_get_embedding_dimension() -> None:
    model = FakeEmbeddingModel()

    chunks = [
        TextChunk(
            chunk_id="lecture_1_p1_c1",
            file_name="lecture_1.pdf",
            page_number=1,
            text="Some text",
        )
    ]

    embedded_chunks = embed_chunks(chunks=chunks, model=model)

    assert get_embedding_dimension(embedded_chunks) == 2


def test_generate_embeddings_detects_wrong_number_of_embeddings() -> None:
    model = BadCountEmbeddingModel()

    with pytest.raises(EmbeddingGenerationError):
        generate_embeddings(texts=["text one", "text two"], model=model)
