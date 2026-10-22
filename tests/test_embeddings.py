import numpy as np
import pytest

from src.embeddings import (
    EmbeddingGenerationError,
    EmbeddingModelLoadError,
    FastEmbedModel,
    embed_chunks,
    generate_embeddings,
    get_embedding_dimension,
    load_embedding_model,
    resolve_model_name,
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
        return [[float(len(sentence)), float(index)] for index, sentence in enumerate(sentences)]


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


class InvalidVectorOutputModel:
    """
    Fake model that returns invalid vector output.
    """

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> list[str]:
        return ["not-a-vector" for _ in sentences]


class NonNumericVectorOutputModel:
    """
    Fake model that returns a vector containing a non-numeric value.
    """

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> list[list[object]]:
        return [[1.0, "not-a-number"] for _ in sentences]


def test_generate_embeddings_rejects_invalid_vector_output() -> None:
    model = InvalidVectorOutputModel()

    with pytest.raises(EmbeddingGenerationError):
        generate_embeddings(
            texts=["Some valid text."],
            model=model,
        )


def test_generate_embeddings_rejects_non_numeric_vector_values() -> None:
    model = NonNumericVectorOutputModel()

    with pytest.raises(EmbeddingGenerationError):
        generate_embeddings(
            texts=["Some valid text."],
            model=model,
        )


class FakeTextEmbedding:
    """
    Stand-in for fastembed.TextEmbedding that returns fixed, unnormalised vectors.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed(self, documents: list[str], batch_size: int = 256):
        for _ in documents:
            yield np.array([3.0, 4.0])


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
        (" all-MiniLM-L6-v2 ", "sentence-transformers/all-MiniLM-L6-v2"),
        ("BAAI/bge-small-en-v1.5", "BAAI/bge-small-en-v1.5"),
    ],
)
def test_resolve_model_name_adds_default_namespace(model_name: str, expected: str) -> None:
    assert resolve_model_name(model_name) == expected


def test_fastembed_model_returns_normalised_vectors(monkeypatch) -> None:
    monkeypatch.setattr("fastembed.TextEmbedding", FakeTextEmbedding)

    model = FastEmbedModel("all-MiniLM-L6-v2")
    embeddings = generate_embeddings(texts=["first", "second"], model=model)

    assert model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert embeddings == [[0.6, 0.8], [0.6, 0.8]]


def test_fastembed_model_can_skip_normalisation(monkeypatch) -> None:
    monkeypatch.setattr("fastembed.TextEmbedding", FakeTextEmbedding)

    model = FastEmbedModel("all-MiniLM-L6-v2")

    assert model.encode(["text"], normalize_embeddings=False) == [[3.0, 4.0]]


def test_load_embedding_model_rejects_empty_name() -> None:
    with pytest.raises(EmbeddingModelLoadError):
        load_embedding_model("   ")


def test_load_embedding_model_wraps_loading_errors(monkeypatch) -> None:
    def failing_text_embedding(model_name: str):
        raise ValueError(f"Unknown model {model_name}")

    monkeypatch.setattr("fastembed.TextEmbedding", failing_text_embedding)

    with pytest.raises(EmbeddingModelLoadError):
        load_embedding_model("not-a-real-model")
