import pytest

from src.models import RetrievedChunk
from src.vector_store import (
    InvalidCollectionNameError,
    InvalidEmbeddingError,
    VectorStoreError,
    filter_chunks_by_max_distance,
    get_chroma_client,
    get_collection_count,
    get_collection_preview,
    get_or_create_collection,
    retrieve_similar_chunks,
    upsert_embedded_chunks,
    validate_collection_name,
)
from tests.helpers import make_embedded_chunk


def test_upsert_embedded_chunks_stores_records(tmp_path) -> None:
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "test_notes")

    chunks = [
        make_embedded_chunk(
            chunk_id="chunk_1",
            text="This is about recursion.",
            embedding=[1.0, 0.0, 0.0],
        ),
        make_embedded_chunk(
            chunk_id="chunk_2",
            text="This is about databases.",
            embedding=[0.0, 1.0, 0.0],
        ),
    ]

    upserted_count = upsert_embedded_chunks(collection, chunks)

    assert upserted_count == 2
    assert get_collection_count(collection) == 2


def test_upsert_embedded_chunks_updates_existing_record(tmp_path) -> None:
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "test_notes")

    original_chunk = make_embedded_chunk(
        chunk_id="chunk_1",
        text="Original text.",
        embedding=[1.0, 0.0],
    )

    updated_chunk = make_embedded_chunk(
        chunk_id="chunk_1",
        text="Updated text.",
        embedding=[0.9, 0.1],
    )

    upsert_embedded_chunks(collection, [original_chunk])
    upsert_embedded_chunks(collection, [updated_chunk])

    assert get_collection_count(collection) == 1

    preview = get_collection_preview(collection, limit=1)

    assert len(preview) == 1
    assert preview[0].chunk_id == "chunk_1"
    assert preview[0].text == "Updated text."


def test_get_collection_preview_returns_metadata(tmp_path) -> None:
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "test_notes")

    chunk = make_embedded_chunk(
        chunk_id="chunk_1",
        text="This chunk discusses binary search.",
        embedding=[1.0, 0.0],
        file_name="algorithms.pdf",
        page_number=7,
    )

    upsert_embedded_chunks(collection, [chunk])

    preview = get_collection_preview(collection, limit=5)

    assert len(preview) == 1
    assert preview[0].chunk_id == "chunk_1"
    assert preview[0].file_name == "algorithms.pdf"
    assert preview[0].page_number == 7
    assert "binary search" in preview[0].text


def test_retrieve_similar_chunks_returns_nearest_chunk(tmp_path) -> None:
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "test_notes")

    chunks = [
        make_embedded_chunk(
            chunk_id="recursion_chunk",
            text="Recursion is when a function calls itself.",
            embedding=[1.0, 0.0, 0.0],
            file_name="algorithms.pdf",
            page_number=3,
        ),
        make_embedded_chunk(
            chunk_id="database_chunk",
            text="A database stores structured information.",
            embedding=[0.0, 1.0, 0.0],
            file_name="databases.pdf",
            page_number=5,
        ),
    ]

    upsert_embedded_chunks(collection, chunks)

    retrieved = retrieve_similar_chunks(
        collection=collection,
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(retrieved) == 1
    assert retrieved[0].chunk_id == "recursion_chunk"
    assert retrieved[0].file_name == "algorithms.pdf"
    assert retrieved[0].page_number == 3
    assert "function calls itself" in retrieved[0].text


def test_upsert_rejects_inconsistent_embedding_dimensions(tmp_path) -> None:
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "test_notes")

    chunks = [
        make_embedded_chunk(
            chunk_id="chunk_1",
            text="First chunk.",
            embedding=[1.0, 0.0],
        ),
        make_embedded_chunk(
            chunk_id="chunk_2",
            text="Second chunk.",
            embedding=[1.0, 0.0, 0.0],
        ),
    ]

    with pytest.raises(InvalidEmbeddingError):
        upsert_embedded_chunks(collection, chunks)


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


@pytest.mark.parametrize(
    "collection_name",
    [
        "ab",
        "bad name",
        "_bad",
        "bad_",
        "bad..name",
        "192.168.1.1",
    ],
)
def test_validate_collection_name_rejects_invalid_names(
    collection_name: str,
) -> None:
    with pytest.raises(InvalidCollectionNameError):
        validate_collection_name(collection_name)


def test_validate_collection_name_accepts_valid_name() -> None:
    validate_collection_name("study.notes_2024-v1")
