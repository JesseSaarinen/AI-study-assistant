from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from src.chunker import chunk_pages
from src.embeddings import EmbeddingModel, embed_chunks
from src.models import EmbeddedChunk, PageText, TextChunk
from src.pdf_loader import UploadedPDFLike, extract_text_from_pdfs
from src.vector_store import (
    get_collection_count,
    get_or_create_collection,
    reset_collection,
    upsert_embedded_chunks,
)

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class IngestionResult:
    """
    Outcome of running uploaded PDFs through the ingestion pipeline.

    stopped_reason is set when the pipeline could not reach the vector store,
    for example because the PDFs contained no selectable text.
    """

    pages: list[PageText] = field(default_factory=list)
    extraction_errors: list[str] = field(default_factory=list)
    chunks: list[TextChunk] = field(default_factory=list)
    embedded_chunks: list[EmbeddedChunk] = field(default_factory=list)
    upserted_count: int = 0
    total_count: int = 0
    stopped_reason: str | None = None

    @property
    def completed(self) -> bool:
        return self.stopped_reason is None


def _report(on_progress: ProgressCallback | None, message: str) -> None:
    if on_progress is not None:
        on_progress(message)


def ingest_documents(
    uploaded_files: Iterable[UploadedPDFLike],
    embedding_model: EmbeddingModel,
    client: Any,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    clear_existing_collection: bool = False,
    max_file_size_mb: int | None = None,
    batch_size: int = 32,
    on_progress: ProgressCallback | None = None,
) -> IngestionResult:
    """
    Run the full ingestion pipeline: extract, chunk, embed, and store.

    This function has no UI dependencies, so the Streamlit app, the evaluation
    script, and the tests all share the same code path.

    When clear_existing_collection is True, the collection is only reset once
    new embeddings are ready, so a failed upload never wipes the stored notes.

    Raises:
        ChunkingError: If chunking settings are invalid.
        EmbeddingError: If embeddings cannot be generated.
        VectorStoreError: If chunks cannot be stored.
    """

    _report(on_progress, "Step 1/4: Extracting text from PDFs...")
    pages, errors = extract_text_from_pdfs(
        uploaded_files=uploaded_files,
        max_file_size_mb=max_file_size_mb,
    )

    if not pages:
        return IngestionResult(
            extraction_errors=errors,
            stopped_reason="No selectable text was extracted from the uploaded PDFs.",
        )

    _report(on_progress, "Step 2/4: Creating text chunks...")
    chunks = chunk_pages(
        pages=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        return IngestionResult(
            pages=pages,
            extraction_errors=errors,
            stopped_reason="No chunks were created from the extracted text.",
        )

    _report(on_progress, "Step 3/4: Generating embeddings...")
    embedded_chunks = embed_chunks(
        chunks=chunks,
        model=embedding_model,
        batch_size=batch_size,
    )

    if not embedded_chunks:
        return IngestionResult(
            pages=pages,
            extraction_errors=errors,
            chunks=chunks,
            stopped_reason="No embeddings were generated.",
        )

    _report(on_progress, "Step 4/4: Storing embeddings in ChromaDB...")
    if clear_existing_collection:
        reset_collection(client=client, collection_name=collection_name)

    collection = get_or_create_collection(client=client, collection_name=collection_name)
    upserted_count = upsert_embedded_chunks(
        collection=collection,
        embedded_chunks=embedded_chunks,
    )

    return IngestionResult(
        pages=pages,
        extraction_errors=errors,
        chunks=chunks,
        embedded_chunks=embedded_chunks,
        upserted_count=upserted_count,
        total_count=get_collection_count(collection),
    )
