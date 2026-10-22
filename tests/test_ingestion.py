from src.ingestion import ingest_documents
from src.vector_store import get_chroma_client, get_or_create_collection, upsert_embedded_chunks
from tests.helpers import (
    FakeUploadedFile,
    KeywordFakeEmbeddingModel,
    create_pdf_with_pages,
    make_embedded_chunk,
)


def make_notes_pdf() -> FakeUploadedFile:
    return FakeUploadedFile(
        "notes.pdf",
        create_pdf_with_pages(
            [
                "Recursion is when a function calls itself with a base case.",
                "A relational database stores data in tables queried with SQL.",
            ]
        ),
    )


def test_ingest_documents_stores_all_chunks(tmp_path) -> None:
    client = get_chroma_client(tmp_path / "chroma")
    progress: list[str] = []

    result = ingest_documents(
        uploaded_files=[make_notes_pdf()],
        embedding_model=KeywordFakeEmbeddingModel(),
        client=client,
        collection_name="ingest_notes",
        chunk_size=500,
        chunk_overlap=50,
        on_progress=progress.append,
    )

    assert result.completed
    assert len(result.pages) == 2
    assert len(result.chunks) == 2
    assert len(result.embedded_chunks) == 2
    assert result.upserted_count == 2
    assert result.total_count == 2
    assert len(progress) == 4
    assert progress[0].startswith("Step 1/4")


def test_ingest_documents_stops_when_no_text_is_extracted(tmp_path) -> None:
    client = get_chroma_client(tmp_path / "chroma")

    result = ingest_documents(
        uploaded_files=[FakeUploadedFile("not_a_pdf.txt", b"plain text")],
        embedding_model=KeywordFakeEmbeddingModel(),
        client=client,
        collection_name="ingest_notes",
        chunk_size=500,
        chunk_overlap=50,
    )

    assert not result.completed
    assert result.stopped_reason
    assert len(result.extraction_errors) == 1
    assert result.upserted_count == 0


def test_failed_ingestion_keeps_existing_collection(tmp_path) -> None:
    client = get_chroma_client(tmp_path / "chroma")
    collection = get_or_create_collection(client, "ingest_notes")
    upsert_embedded_chunks(
        collection=collection,
        embedded_chunks=[make_embedded_chunk("old_chunk", "Old notes.", [1.0, 0.0])],
    )

    result = ingest_documents(
        uploaded_files=[FakeUploadedFile("not_a_pdf.txt", b"plain text")],
        embedding_model=KeywordFakeEmbeddingModel(),
        client=client,
        collection_name="ingest_notes",
        chunk_size=500,
        chunk_overlap=50,
        clear_existing_collection=True,
    )

    assert not result.completed
    assert get_or_create_collection(client, "ingest_notes").count() == 1


def test_ingest_documents_can_replace_existing_collection(tmp_path) -> None:
    client = get_chroma_client(tmp_path / "chroma")
    collection = get_or_create_collection(client, "ingest_notes")
    upsert_embedded_chunks(
        collection=collection,
        embedded_chunks=[make_embedded_chunk("old_chunk", "Old notes.", [1.0, 0.0])],
    )

    result = ingest_documents(
        uploaded_files=[make_notes_pdf()],
        embedding_model=KeywordFakeEmbeddingModel(),
        client=client,
        collection_name="ingest_notes",
        chunk_size=500,
        chunk_overlap=50,
        clear_existing_collection=True,
    )

    assert result.completed
    assert result.total_count == 2
