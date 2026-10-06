from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from sentence_transformers import SentenceTransformer

from src.models import EmbeddedChunk, TextChunk


class EmbeddingModel(Protocol):
    """
    Minimal interface needed from a Sentence Transformers style model.

    This allows us to test embedding logic with a fake model without downloading
    the real model during pytest.
    """

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> Any:
        ...


class EmbeddingError(Exception):
    """
    Base exception for embedding-related errors.
    """


class EmbeddingModelLoadError(EmbeddingError):
    """
    Raised when the embedding model cannot be loaded.
    """


class EmbeddingGenerationError(EmbeddingError):
    """
    Raised when embeddings cannot be generated correctly.
    """


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Load a Sentence Transformers model.

    The first time this runs, the model may be downloaded from Hugging Face.
    Later runs usually load from the local cache.

    Args:
        model_name: Name of the Sentence Transformers model.

    Returns:
        A loaded SentenceTransformer model.

    Raises:
        EmbeddingModelLoadError: If the model name is missing or loading fails.
    """

    cleaned_model_name = model_name.strip()

    if not cleaned_model_name:
        raise EmbeddingModelLoadError("Embedding model name cannot be empty.")

    try:
        return SentenceTransformer(cleaned_model_name)
    except Exception as exc:
        raise EmbeddingModelLoadError(
            f"Could not load embedding model '{cleaned_model_name}'. "
            "Check your internet connection on first run and confirm the model name is valid."
        ) from exc


def _vector_to_float_list(vector: Any) -> list[float]:
    """
    Convert one embedding vector into a plain list of floats.
    """

    if hasattr(vector, "tolist"):
        vector = vector.tolist()

    if not isinstance(vector, Sequence) or isinstance(vector, str | bytes):
        raise EmbeddingGenerationError(
            "Embedding model returned an invalid vector format."
        )

    try:
        float_vector = [float(value) for value in vector]
    except Exception as exc:
        raise EmbeddingGenerationError(
            "Embedding vector contained values that could not be converted to floats."
        ) from exc

    if not float_vector:
        raise EmbeddingGenerationError("Embedding vector cannot be empty.")

    return float_vector


def _embedding_output_to_lists(raw_embeddings: Any) -> list[list[float]]:
    """
    Convert model output into a list of float vectors.

    Sentence Transformers usually returns a NumPy array, but fake test models may
    return normal Python lists. This function handles both.
    """

    if hasattr(raw_embeddings, "tolist"):
        raw_embeddings = raw_embeddings.tolist()

    if not isinstance(raw_embeddings, Sequence) or isinstance(
        raw_embeddings, str | bytes
    ):
        raise EmbeddingGenerationError(
            "Embedding model returned an invalid embeddings format."
        )

    return [_vector_to_float_list(vector) for vector in raw_embeddings]


def generate_embeddings(
    texts: Iterable[str],
    model: EmbeddingModel,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    show_progress_bar: bool = False,
) -> list[list[float]]:
    """
    Generate embeddings for a collection of text strings.

    Args:
        texts: Text values to embed.
        model: Sentence Transformers compatible model.
        batch_size: Number of texts to process per batch.
        normalize_embeddings: Whether to normalise embeddings to unit length.
        show_progress_bar: Whether to show the Sentence Transformers progress bar.

    Returns:
        A list of embedding vectors.

    Raises:
        EmbeddingGenerationError: If input is invalid or model output is invalid.
    """

    if batch_size <= 0:
        raise EmbeddingGenerationError("Embedding batch_size must be greater than 0.")

    text_list = [text.strip() for text in texts]

    if not text_list:
        return []

    if any(not text for text in text_list):
        raise EmbeddingGenerationError("Cannot generate embeddings for empty text.")

    try:
        raw_embeddings = model.encode(
            text_list,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
        )
    except Exception as exc:
        raise EmbeddingGenerationError("Failed to generate embeddings.") from exc

    embeddings = _embedding_output_to_lists(raw_embeddings)

    if len(embeddings) != len(text_list):
        raise EmbeddingGenerationError(
            "Embedding model returned a different number of embeddings than input texts."
        )

    return embeddings


def embed_chunks(
    chunks: list[TextChunk],
    model: EmbeddingModel,
    batch_size: int = 32,
) -> list[EmbeddedChunk]:
    """
    Generate embeddings for text chunks while preserving chunk metadata.

    Args:
        chunks: Text chunks to embed.
        model: Sentence Transformers compatible model.
        batch_size: Number of chunks to process per batch.

    Returns:
        A list of EmbeddedChunk objects.
    """

    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]

    embeddings = generate_embeddings(
        texts=texts,
        model=model,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return [
        EmbeddedChunk(chunk=chunk, embedding=embedding)
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]


def get_embedding_dimension(embedded_chunks: list[EmbeddedChunk]) -> int:
    """
    Return the embedding vector dimension.

    For all-MiniLM-L6-v2, this should usually be 384.
    """

    if not embedded_chunks:
        return 0

    return len(embedded_chunks[0].embedding)
