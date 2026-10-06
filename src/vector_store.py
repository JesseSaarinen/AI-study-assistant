import re
from collections.abc import Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from src.models import EmbeddedChunk, RetrievedChunk


DEFAULT_COLLECTION_NAME = "study_notes"


class VectorStoreError(Exception):
    """
    Base exception for vector database errors.
    """


class InvalidCollectionNameError(VectorStoreError):
    """
    Raised when the ChromaDB collection name is invalid.
    """


class InvalidEmbeddingError(VectorStoreError):
    """
    Raised when embeddings are missing or malformed.
    """


def validate_collection_name(collection_name: str) -> None:
    """
    Validate a ChromaDB collection name.

    ChromaDB collection names must be 3-512 characters and generally contain
    letters, numbers, underscores, hyphens, or dots.
    """

    if not collection_name or not collection_name.strip():
        raise InvalidCollectionNameError("ChromaDB collection name cannot be empty.")

    collection_name = collection_name.strip()

    if len(collection_name) < 3 or len(collection_name) > 512:
        raise InvalidCollectionNameError(
            "ChromaDB collection name must be between 3 and 512 characters."
        )

    pattern = r"^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$"

    if not re.match(pattern, collection_name):
        raise InvalidCollectionNameError(
            "ChromaDB collection name must start and end with a letter or number "
            "and may only contain letters, numbers, dots, underscores, and hyphens."
        )

    if ".." in collection_name:
        raise InvalidCollectionNameError(
            "ChromaDB collection name cannot contain two consecutive dots."
        )

    if re.match(r"^\d+\.\d+\.\d+\.\d+$", collection_name):
        raise InvalidCollectionNameError(
            "ChromaDB collection name cannot be an IPv4 address."
        )


def get_chroma_client(chroma_db_dir: str | Path) -> chromadb.PersistentClient:
    """
    Create a persistent ChromaDB client.

    Args:
        chroma_db_dir: Directory where ChromaDB should store its local database.

    Returns:
        A persistent ChromaDB client.
    """

    db_path = Path(chroma_db_dir)
    db_path.mkdir(parents=True, exist_ok=True)

    try:
        return chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False),
        )
    except Exception as exc:
        raise VectorStoreError(
            f"Could not initialise ChromaDB at '{db_path}'."
        ) from exc


def get_or_create_collection(
    client: chromadb.PersistentClient,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Any:
    """
    Get or create a ChromaDB collection.

    We use cosine distance because our embeddings represent semantic direction.
    Sentence Transformers embeddings are also normalised in Phase 4.
    """

    validate_collection_name(collection_name)

    try:
        return client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        raise VectorStoreError(
            f"Could not get or create ChromaDB collection '{collection_name}'."
        ) from exc


def reset_collection(
    client: chromadb.PersistentClient,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> None:
    """
    Delete a collection if it exists.

    This is useful in the Streamlit app because a user may upload a new set of
    PDFs and we do not want old notes mixed with new notes.
    """

    validate_collection_name(collection_name)

    try:
        client.delete_collection(name=collection_name)
    except Exception as exc:
        message = str(exc).lower()

        if "does not exist" in message or "not found" in message:
            return

        raise VectorStoreError(
            f"Could not reset ChromaDB collection '{collection_name}'."
        ) from exc


def _validate_embedding_vector(vector: Sequence[float], chunk_id: str) -> list[float]:
    """
    Validate and convert one embedding vector to a plain list of floats.
    """

    if not isinstance(vector, Sequence) or isinstance(vector, str | bytes):
        raise InvalidEmbeddingError(
            f"Embedding for chunk '{chunk_id}' is not a valid sequence."
        )

    if not vector:
        raise InvalidEmbeddingError(f"Embedding for chunk '{chunk_id}' is empty.")

    float_vector: list[float] = []

    for value in vector:
        if not isinstance(value, Real):
            raise InvalidEmbeddingError(
                f"Embedding for chunk '{chunk_id}' contains a non-numeric value."
            )

        float_vector.append(float(value))

    return float_vector


def _validate_embedded_chunks(embedded_chunks: list[EmbeddedChunk]) -> list[list[float]]:
    """
    Validate embedded chunks before inserting into ChromaDB.

    Returns:
        Validated embedding vectors as lists of floats.
    """

    validated_embeddings: list[list[float]] = []

    expected_dimension: int | None = None

    for embedded_chunk in embedded_chunks:
        chunk = embedded_chunk.chunk
        embedding = _validate_embedding_vector(
            vector=embedded_chunk.embedding,
            chunk_id=chunk.chunk_id,
        )

        if expected_dimension is None:
            expected_dimension = len(embedding)

        if len(embedding) != expected_dimension:
            raise InvalidEmbeddingError(
                "All embeddings must have the same dimension. "
                f"Expected {expected_dimension}, got {len(embedding)} "
                f"for chunk '{chunk.chunk_id}'."
            )

        if not chunk.chunk_id.strip():
            raise InvalidEmbeddingError("Chunk ID cannot be empty.")

        if not chunk.text.strip():
            raise InvalidEmbeddingError(
                f"Chunk '{chunk.chunk_id}' has empty text."
            )

        validated_embeddings.append(embedding)

    return validated_embeddings


def upsert_embedded_chunks(
    collection: Any,
    embedded_chunks: list[EmbeddedChunk],
) -> int:
    """
    Insert or update embedded chunks in ChromaDB.

    ChromaDB stores:
    - ids: stable chunk IDs
    - documents: original chunk text
    - embeddings: numerical vectors
    - metadatas: source information such as file name and page number

    Args:
        collection: ChromaDB collection.
        embedded_chunks: Embedded chunks to store.

    Returns:
        Number of chunks upserted.
    """

    if not embedded_chunks:
        return 0

    embeddings = _validate_embedded_chunks(embedded_chunks)

    ids = [embedded.chunk.chunk_id for embedded in embedded_chunks]
    documents = [embedded.chunk.text for embedded in embedded_chunks]

    metadatas = [
        {
            "chunk_id": embedded.chunk.chunk_id,
            "file_name": embedded.chunk.file_name,
            "page_number": embedded.chunk.page_number,
            "text_length": len(embedded.chunk.text),
        }
        for embedded in embedded_chunks
    ]

    try:
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    except Exception as exc:
        raise VectorStoreError("Failed to upsert embedded chunks into ChromaDB.") from exc

    return len(embedded_chunks)


def get_collection_count(collection: Any) -> int:
    """
    Return the number of records in a ChromaDB collection.
    """

    try:
        return int(collection.count())
    except Exception as exc:
        raise VectorStoreError("Failed to count ChromaDB collection records.") from exc


def _page_number_from_metadata(metadata: dict[str, Any]) -> int:
    """
    Safely parse page number metadata.
    """

    raw_page_number = metadata.get("page_number", 0)

    try:
        return int(raw_page_number)
    except (TypeError, ValueError):
        return 0


def _record_to_retrieved_chunk(
    record_id: str,
    document: str | None,
    metadata: dict[str, Any] | None,
    distance: float | None = None,
) -> RetrievedChunk:
    """
    Convert a ChromaDB record into a RetrievedChunk dataclass.
    """

    metadata = metadata or {}

    return RetrievedChunk(
        chunk_id=str(metadata.get("chunk_id") or record_id),
        file_name=str(metadata.get("file_name") or "unknown"),
        page_number=_page_number_from_metadata(metadata),
        text=document or "",
        distance=distance,
    )


def get_collection_preview(
    collection: Any,
    limit: int = 5,
) -> list[RetrievedChunk]:
    """
    Fetch a small preview of stored chunks from ChromaDB.

    This is mainly used by the Streamlit UI to confirm that storage worked.
    """

    if limit <= 0:
        return []

    try:
        results = collection.get(
            limit=limit,
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        raise VectorStoreError("Failed to fetch ChromaDB preview records.") from exc

    ids = results.get("ids") or []
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    preview: list[RetrievedChunk] = []

    for index, record_id in enumerate(ids):
        document = documents[index] if index < len(documents) else ""
        metadata = metadatas[index] if index < len(metadatas) else {}

        preview.append(
            _record_to_retrieved_chunk(
                record_id=record_id,
                document=document,
                metadata=metadata,
            )
        )

    return preview

def filter_chunks_by_max_distance(
    chunks: list[RetrievedChunk],
    max_distance: float | None,
) -> list[RetrievedChunk]:
    """
    Filter retrieved chunks by maximum allowed distance.

    ChromaDB cosine distance is lower when chunks are more similar.
    If max_distance is None, filtering is disabled.

    Args:
        chunks: Retrieved chunks from ChromaDB.
        max_distance: Maximum allowed distance, or None to disable filtering.

    Returns:
        Chunks whose distance is less than or equal to max_distance.
    """

    if max_distance is None:
        return chunks

    if max_distance < 0:
        raise VectorStoreError("max_distance cannot be negative.")

    return [
        chunk
        for chunk in chunks
        if chunk.distance is not None and chunk.distance <= max_distance
    ]

def retrieve_similar_chunks(
    collection: Any,
    query_embedding: list[float],
    top_k: int = 5,
    max_distance: float | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve the most similar chunks for a query embedding.

    Args:
        collection: ChromaDB collection.
        query_embedding: Embedding vector for the user's question.
        top_k: Number of chunks to retrieve.
        max_distance: Optional maximum allowed ChromaDB distance.

    Returns:
        A list of RetrievedChunk objects.
    """

    if top_k <= 0:
        raise VectorStoreError("top_k must be greater than 0.")

    if max_distance is not None and max_distance < 0:
        raise VectorStoreError("max_distance cannot be negative.")

    if not query_embedding:
        raise InvalidEmbeddingError("Query embedding cannot be empty.")

    collection_count = get_collection_count(collection)

    if collection_count == 0:
        return []

    n_results = min(top_k, collection_count)

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise VectorStoreError("Failed to query ChromaDB collection.") from exc

    ids_nested = results.get("ids") or [[]]
    documents_nested = results.get("documents") or [[]]
    metadatas_nested = results.get("metadatas") or [[]]
    distances_nested = results.get("distances") or [[]]

    ids = ids_nested[0] if ids_nested else []
    documents = documents_nested[0] if documents_nested else []
    metadatas = metadatas_nested[0] if metadatas_nested else []
    distances = distances_nested[0] if distances_nested else []

    retrieved_chunks: list[RetrievedChunk] = []

    for index, record_id in enumerate(ids):
        document = documents[index] if index < len(documents) else ""
        metadata = metadatas[index] if index < len(metadatas) else {}
        raw_distance = distances[index] if index < len(distances) else None
        distance = float(raw_distance) if raw_distance is not None else None

        retrieved_chunks.append(
            _record_to_retrieved_chunk(
                record_id=record_id,
                document=document,
                metadata=metadata,
                distance=distance,
            )
        )

    return filter_chunks_by_max_distance(
        chunks=retrieved_chunks,
        max_distance=max_distance,
    )