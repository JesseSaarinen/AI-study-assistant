import pytest

from src.models import EmbeddedChunk, RetrievedChunk, TextChunk
from src.vector_store import (
    VectorStoreError,
    filter_chunks_by_max_distance,
    get_chroma_client,
    get_or_create_collection,
    retrieve_similar_chunks,
    upsert_embedded_chunks,
)


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


def test_filter_chunks_by_max_distance_returns_all_when_disabled() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk_1",
            file_name="lecture.pdf",
            page_number=1,
            text="Text 1",
            distance=0.2,
        ),
        RetrievedChunk(
            chunk_id="chunk_2",
            file_name="lecture.pdf",
            page_number=2,
            text="Text 2",
            distance=1.2,
        ),
    ]

    filtered = filter_chunks_by_max_distance(chunks, max_distance=None)

    assert filtered == chunks


def test_filter_chunks_by_max_distance_filters_weak_matches() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk_1",
            file_name="lecture.pdf",
            page_number=1,
            text="Text 1",
            distance=0.2,
        ),
        RetrievedChunk(
            chunk_id="chunk_2",
            file_name="lecture.pdf",
            page_number=2,
            text="Text 2",
            distance=1.2,
        ),
    ]

    filtered = filter_chunks_by_max_distance(chunks, max_distance=0.5)

    assert len(filtered) == 1
    assert filtered[0].chunk_id == "chunk_1"


def test_filter_chunks_by_max_distance_rejects_negative_threshold() -> None:
    with pytest.raises(VectorStoreError):
        filter_chunks_by_max_distance([], max_distance=-0.1)


def test_retrieve_similar_chunks_applies_max_distance(tmp_path) -> None:
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "robust_notes")

    chunk = make_embedded_chunk(
        chunk_id="recursion_chunk",
        text="Recursion is when a function calls itself.",
        embedding=[1.0, 0.0],
    )

    upsert_embedded_chunks(collection, [chunk])

    retrieved = retrieve_similar_chunks(
        collection=collection,
        query_embedding=[0.0, 1.0],
        top_k=1,
        max_distance=0.5,
    )

    assert retrieved == []
