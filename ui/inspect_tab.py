import streamlit as st

from src.config import AppConfig
from src.embeddings import get_embedding_dimension
from src.models import EmbeddedChunk, PageText, TextChunk
from src.ui_helpers import build_pipeline_status_items
from src.vector_store import (
    VectorStoreError,
    get_collection_count,
    get_collection_preview,
    get_or_create_collection,
)
from ui.resources import get_cached_chroma_client


def get_safe_stored_count(chroma_db_dir: str, collection_name: str) -> int:
    """
    Safely return the number of stored chunks in ChromaDB.

    If ChromaDB cannot be read, return 0 so the UI does not crash.
    """

    try:
        client = get_cached_chroma_client(chroma_db_dir)
        collection = get_or_create_collection(client, collection_name)
        return get_collection_count(collection)
    except VectorStoreError:
        return 0


def display_pipeline_overview(
    pages: list[PageText],
    chunks: list[TextChunk],
    embedded_chunks: list[EmbeddedChunk],
    stored_count: int,
) -> None:
    """
    Display a compact pipeline status checklist.
    """

    st.subheader("Pipeline overview")

    status_items = build_pipeline_status_items(
        page_count=len(pages),
        chunk_count=len(chunks),
        embedded_count=len(embedded_chunks),
        stored_count=stored_count,
    )

    for item in status_items:
        st.markdown(item)


def display_extraction_errors(errors: list[str]) -> None:
    """
    Display PDF extraction warnings.
    """

    if not errors:
        return

    st.subheader("PDF extraction warnings")

    for error in errors:
        st.warning(error)


def display_extraction_summary(pages: list[PageText]) -> None:
    """
    Display summary metrics and a preview of extracted PDF text.
    """

    if not pages:
        st.info("No extracted PDF text is currently available.")
        return

    total_pages = len(pages)
    total_characters = sum(len(page.text) for page in pages)
    unique_files = sorted({page.file_name for page in pages})

    st.subheader("Extracted text summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Files with text", len(unique_files))

    with col2:
        st.metric("Pages with text", total_pages)

    with col3:
        st.metric("Characters extracted", f"{total_characters:,}")

    page_labels = [
        f"{index + 1}. {page.file_name} — page {page.page_number}"
        for index, page in enumerate(pages)
    ]

    selected_label = st.selectbox(
        "Choose a page to preview",
        options=page_labels,
        key="page_preview_select",
    )

    selected_index = page_labels.index(selected_label)
    selected_page = pages[selected_index]

    st.caption(f"Source: `{selected_page.file_name}`, page `{selected_page.page_number}`")

    st.text_area(
        "Extracted page text",
        value=selected_page.text,
        height=300,
        disabled=True,
    )


def display_chunk_summary(chunks: list[TextChunk]) -> None:
    """
    Display summary metrics and a preview of generated text chunks.
    """

    if not chunks:
        st.info("No text chunks are currently available.")
        return

    total_chunks = len(chunks)
    total_characters = sum(len(chunk.text) for chunk in chunks)
    average_chunk_size = round(total_characters / total_chunks)
    unique_pages = {(chunk.file_name, chunk.page_number) for chunk in chunks}

    st.subheader("Text chunk summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Chunks created", total_chunks)

    with col2:
        st.metric("Source pages", len(unique_pages))

    with col3:
        st.metric("Average chunk size", f"{average_chunk_size:,} chars")

    chunk_labels = [
        (f"{index + 1}. {chunk.file_name} — page {chunk.page_number} — {chunk.chunk_id}")
        for index, chunk in enumerate(chunks)
    ]

    selected_label = st.selectbox(
        "Choose a chunk to preview",
        options=chunk_labels,
        key="chunk_preview_select",
    )

    selected_index = chunk_labels.index(selected_label)
    selected_chunk = chunks[selected_index]

    st.caption(f"Chunk ID: `{selected_chunk.chunk_id}`")
    st.caption(f"Source: `{selected_chunk.file_name}`, page `{selected_chunk.page_number}`")

    st.text_area(
        "Chunk text",
        value=selected_chunk.text,
        height=300,
        disabled=True,
    )


def display_embedding_summary(
    embedded_chunks: list[EmbeddedChunk],
    model_name: str,
) -> None:
    """
    Display summary metrics and a preview of generated embeddings.
    """

    if not embedded_chunks:
        st.info("No embeddings are currently available.")
        return

    embedding_dimension = get_embedding_dimension(embedded_chunks)

    st.subheader("Embedding summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Embedded chunks", len(embedded_chunks))

    with col2:
        st.metric("Embedding dimension", embedding_dimension)

    with col3:
        st.metric("Embedding model", model_name)

    embedding_labels = [
        (
            f"{index + 1}. {embedded.chunk.file_name} — "
            f"page {embedded.chunk.page_number} — {embedded.chunk.chunk_id}"
        )
        for index, embedded in enumerate(embedded_chunks)
    ]

    selected_label = st.selectbox(
        "Choose an embedded chunk to preview",
        options=embedding_labels,
        key="embedding_preview_select",
    )

    selected_index = embedding_labels.index(selected_label)
    selected_embedded_chunk = embedded_chunks[selected_index]

    chunk = selected_embedded_chunk.chunk
    embedding = selected_embedded_chunk.embedding

    st.caption(f"Chunk ID: `{chunk.chunk_id}`")
    st.caption(f"Source: `{chunk.file_name}`, page `{chunk.page_number}`")
    st.caption(f"Full embedding length: `{len(embedding)}`")

    st.write("First 10 embedding values:")

    st.code(
        str([round(value, 5) for value in embedding[:10]]),
        language="python",
    )

    st.text_area(
        "Original chunk text",
        value=chunk.text,
        height=220,
        disabled=True,
    )


def display_vector_store_summary(
    chroma_db_dir: str,
    collection_name: str,
) -> int:
    """
    Display ChromaDB collection status and a preview of stored chunks.

    Returns:
        Number of stored chunks.
    """

    try:
        client = get_cached_chroma_client(chroma_db_dir)
        collection = get_or_create_collection(client, collection_name)
        stored_count = get_collection_count(collection)
    except VectorStoreError as exc:
        st.warning(f"Could not read ChromaDB status: {exc}")
        return 0

    if stored_count == 0:
        st.info("No chunks are currently stored in ChromaDB.")
        return 0

    st.subheader("Vector database summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Stored chunks", stored_count)

    with col2:
        st.metric("Collection", collection_name)

    with col3:
        st.metric("Database", chroma_db_dir)

    try:
        preview_records = get_collection_preview(collection, limit=5)
    except VectorStoreError as exc:
        st.warning(f"Could not load ChromaDB preview: {exc}")
        return stored_count

    if not preview_records:
        return stored_count

    record_labels = [
        (f"{index + 1}. {record.file_name} — page {record.page_number} — {record.chunk_id}")
        for index, record in enumerate(preview_records)
    ]

    selected_label = st.selectbox(
        "Choose a stored record to preview",
        options=record_labels,
        key="vector_store_preview_select",
    )

    selected_index = record_labels.index(selected_label)
    selected_record = preview_records[selected_index]

    st.caption(f"Chunk ID: `{selected_record.chunk_id}`")
    st.caption(f"Source: `{selected_record.file_name}`, page `{selected_record.page_number}`")

    st.text_area(
        "Stored chunk text",
        value=selected_record.text,
        height=220,
        disabled=True,
    )

    return stored_count


def display_inspect_tab(config: AppConfig) -> int:
    """
    Display pipeline inspection panels.

    Returns:
        Number of stored chunks in ChromaDB.
    """

    st.header("2. Inspect processed data")

    stored_count = get_safe_stored_count(
        chroma_db_dir=str(config.chroma_db_dir),
        collection_name=config.chroma_collection_name,
    )

    display_pipeline_overview(
        pages=st.session_state.extracted_pages,
        chunks=st.session_state.text_chunks,
        embedded_chunks=st.session_state.embedded_chunks,
        stored_count=stored_count,
    )

    st.divider()

    display_extraction_errors(st.session_state.pdf_extraction_errors)

    with st.expander("PDF extraction details", expanded=False):
        display_extraction_summary(st.session_state.extracted_pages)

    with st.expander("Text chunking details", expanded=False):
        display_chunk_summary(st.session_state.text_chunks)

    with st.expander("Embedding details", expanded=False):
        display_embedding_summary(
            embedded_chunks=st.session_state.embedded_chunks,
            model_name=config.embedding_model_name,
        )

    with st.expander("ChromaDB vector store details", expanded=True):
        stored_count = display_vector_store_summary(
            chroma_db_dir=str(config.chroma_db_dir),
            collection_name=config.chroma_collection_name,
        )

    return stored_count
