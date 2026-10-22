import streamlit as st

from src.chunker import ChunkingError, chunk_pages
from src.config import AppConfig
from src.embeddings import EmbeddingError, embed_chunks
from src.ingestion import ingest_documents
from src.pdf_loader import extract_text_from_pdfs
from src.vector_store import (
    VectorStoreError,
    get_collection_count,
    get_or_create_collection,
    reset_collection,
    upsert_embedded_chunks,
)
from ui.resources import get_cached_chroma_client, get_cached_embedding_model


def run_end_to_end_processing(
    uploaded_files,
    config: AppConfig,
    clear_existing_collection: bool,
) -> None:
    """
    Run the full document processing pipeline from uploaded PDFs to ChromaDB.
    """

    with st.status("Processing PDFs...", expanded=True) as status:
        try:
            result = ingest_documents(
                uploaded_files=uploaded_files,
                embedding_model=get_cached_embedding_model(config.embedding_model_name),
                client=get_cached_chroma_client(str(config.chroma_db_dir)),
                collection_name=config.chroma_collection_name,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                clear_existing_collection=clear_existing_collection,
                max_file_size_mb=config.max_uploaded_file_mb,
                on_progress=st.write,
            )
        except (ChunkingError, EmbeddingError, VectorStoreError):
            status.update(label="Processing failed.", state="error")
            raise

        st.session_state.extracted_pages = result.pages
        st.session_state.pdf_extraction_errors = result.extraction_errors
        st.session_state.text_chunks = result.chunks
        st.session_state.embedded_chunks = result.embedded_chunks
        st.session_state.vector_store_message = ""
        st.session_state.rag_answer = None

        if result.extraction_errors:
            st.write(f"Extraction completed with {len(result.extraction_errors)} warning(s).")

        if not result.completed:
            status.update(
                label=f"Processing stopped: {result.stopped_reason}",
                state="error",
            )
            st.warning(result.stopped_reason)
            return

        st.session_state.vector_store_message = (
            f"Stored {result.upserted_count} chunk(s). "
            f"Collection now contains {result.total_count} chunk(s)."
        )

        status.update(label="Processing complete.", state="complete")
        st.success(st.session_state.vector_store_message)


def display_manual_processing_controls(
    uploaded_files,
    config: AppConfig,
) -> None:
    """
    Display manual step-by-step controls for debugging and learning.
    """

    with st.expander("Advanced: run processing steps manually", expanded=False):
        st.write(
            """
            These controls are useful for debugging or demonstrating each stage
            of the RAG pipeline separately.
            """
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            extract_clicked = st.button(
                "1. Extract text",
                disabled=not uploaded_files,
                key="manual_extract_button",
            )

        with col2:
            chunk_clicked = st.button(
                "2. Create chunks",
                disabled=not st.session_state.extracted_pages,
                key="manual_chunk_button",
            )

        with col3:
            embed_clicked = st.button(
                "3. Generate embeddings",
                disabled=not st.session_state.text_chunks,
                key="manual_embed_button",
            )

        with col4:
            store_clicked = st.button(
                "4. Store in ChromaDB",
                disabled=not st.session_state.embedded_chunks,
                key="manual_store_button",
            )

        if extract_clicked:
            with st.spinner("Extracting text from PDFs..."):
                pages, errors = extract_text_from_pdfs(
                    uploaded_files=uploaded_files,
                    max_file_size_mb=config.max_uploaded_file_mb,
                )

            st.session_state.extracted_pages = pages
            st.session_state.pdf_extraction_errors = errors
            st.session_state.text_chunks = []
            st.session_state.embedded_chunks = []
            st.session_state.vector_store_message = ""
            st.session_state.rag_answer = None

            if pages:
                st.success(f"Extracted text from {len(pages)} page(s).")
            else:
                st.warning("No text was extracted.")

        if chunk_clicked:
            try:
                with st.spinner("Creating chunks..."):
                    chunks = chunk_pages(
                        pages=st.session_state.extracted_pages,
                        chunk_size=config.chunk_size,
                        chunk_overlap=config.chunk_overlap,
                    )

                st.session_state.text_chunks = chunks
                st.session_state.embedded_chunks = []
                st.session_state.vector_store_message = ""
                st.session_state.rag_answer = None

                st.success(f"Created {len(chunks)} chunk(s).")

            except ChunkingError as exc:
                st.error(f"Chunking error: {exc}")

        if embed_clicked:
            try:
                with st.spinner("Generating embeddings..."):
                    embedding_model = get_cached_embedding_model(config.embedding_model_name)

                    embedded_chunks = embed_chunks(
                        chunks=st.session_state.text_chunks,
                        model=embedding_model,
                        batch_size=32,
                    )

                st.session_state.embedded_chunks = embedded_chunks
                st.session_state.vector_store_message = ""
                st.session_state.rag_answer = None

                st.success(f"Generated embeddings for {len(embedded_chunks)} chunk(s).")

            except EmbeddingError as exc:
                st.error(f"Embedding error: {exc}")

        if store_clicked:
            try:
                with st.spinner("Storing embeddings in ChromaDB..."):
                    client = get_cached_chroma_client(str(config.chroma_db_dir))

                    collection = get_or_create_collection(
                        client=client,
                        collection_name=config.chroma_collection_name,
                    )

                    upserted_count = upsert_embedded_chunks(
                        collection=collection,
                        embedded_chunks=st.session_state.embedded_chunks,
                    )

                    total_count = get_collection_count(collection)

                st.session_state.vector_store_message = (
                    f"Stored {upserted_count} chunk(s). "
                    f"Collection now contains {total_count} chunk(s)."
                )
                st.session_state.rag_answer = None

                st.success(st.session_state.vector_store_message)

            except VectorStoreError as exc:
                st.error(f"Vector store error: {exc}")


def display_process_tab(config: AppConfig) -> None:
    """
    Display the upload and processing tab.
    """

    st.header("1. Process notes")

    st.write(
        """
        Upload one or more PDF lecture notes, then process them into searchable
        vector embeddings.
        """
    )

    uploaded_files = st.file_uploader(
        "Upload one or more PDF lecture notes",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write("Uploaded files:")

        for uploaded_file in uploaded_files:
            st.write(f"- `{uploaded_file.name}`")

    else:
        st.info("Upload one or more PDFs to begin.")

    clear_existing_collection = st.checkbox(
        "Clear existing ChromaDB collection before storing",
        value=True,
        help=(
            "Recommended. This prevents notes from previous uploads being mixed "
            "with the current upload."
        ),
    )

    process_clicked = st.button(
        "Process PDFs end-to-end",
        type="primary",
        disabled=not uploaded_files,
    )

    if process_clicked:
        try:
            run_end_to_end_processing(
                uploaded_files=uploaded_files,
                config=config,
                clear_existing_collection=clear_existing_collection,
            )
        except ChunkingError as exc:
            st.error(f"Chunking error: {exc}")
        except EmbeddingError as exc:
            st.error(f"Embedding error: {exc}")
        except VectorStoreError as exc:
            st.error(f"Vector store error: {exc}")
        except Exception as exc:
            st.error(f"Unexpected processing error: {exc}")

    display_manual_processing_controls(
        uploaded_files=uploaded_files,
        config=config,
    )

    if st.session_state.vector_store_message:
        st.success(st.session_state.vector_store_message)

    st.divider()

    with st.expander("Reset vector database collection", expanded=False):
        st.warning(
            """
            This deletes the configured ChromaDB collection from local storage.
            It does not delete your source PDFs.
            """
        )

        if st.button("Delete ChromaDB collection"):
            try:
                client = get_cached_chroma_client(str(config.chroma_db_dir))
                reset_collection(
                    client=client,
                    collection_name=config.chroma_collection_name,
                )

                st.session_state.vector_store_message = ""
                st.session_state.rag_answer = None

                st.success("ChromaDB collection deleted.")

            except VectorStoreError as exc:
                st.error(f"Could not delete collection: {exc}")
