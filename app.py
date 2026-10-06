import streamlit as st

from src.chunker import ChunkingError, chunk_pages
from src.citations import (
    extract_source_numbers,
    format_citation,
    get_citations_to_display,
)
from src.config import AppConfig, ensure_storage_dir, get_config, has_api_key
from src.embeddings import (
    EmbeddingError,
    embed_chunks,
    get_embedding_dimension,
    load_embedding_model,
)
from src.models import EmbeddedChunk, PageText, RagAnswer, TextChunk
from src.pdf_loader import extract_text_from_pdfs
from src.rag_pipeline import (
    ChatModelError,
    EmptyQuestionError,
    OpenAIChatModel,
    RagPipelineError,
    answer_question,
)
from src.ui_helpers import (
    build_pipeline_status_items,
    format_source_label,
    truncate_text,
)
from src.vector_store import (
    VectorStoreError,
    get_chroma_client,
    get_collection_count,
    get_collection_preview,
    get_or_create_collection,
    reset_collection,
    upsert_embedded_chunks,
)


@st.cache_resource(show_spinner=False)
def get_cached_embedding_model(model_name: str):
    """
    Load and cache the embedding model.

    Streamlit reruns the app after interactions. Caching prevents repeatedly
    loading the Sentence Transformers model.
    """

    return load_embedding_model(model_name)


@st.cache_resource(show_spinner=False)
def get_cached_chroma_client(chroma_db_dir: str):
    """
    Load and cache the ChromaDB client.
    """

    return get_chroma_client(chroma_db_dir)


@st.cache_resource(show_spinner=False)
def get_cached_chat_model(api_key: str, base_url: str):
    """
    Load and cache the OpenAI-compatible chat client.
    """

    return OpenAIChatModel(api_key=api_key, base_url=base_url)


def initialise_session_state() -> None:
    """
    Initialise Streamlit session state values used by the app.
    """

    defaults = {
        "extracted_pages": [],
        "pdf_extraction_errors": [],
        "text_chunks": [],
        "embedded_chunks": [],
        "vector_store_message": "",
        "rag_answer": None,
        "last_question": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_working_state() -> None:
    """
    Clear in-memory Streamlit state.

    This does not delete the ChromaDB collection from disk.
    """

    st.session_state.extracted_pages = []
    st.session_state.pdf_extraction_errors = []
    st.session_state.text_chunks = []
    st.session_state.embedded_chunks = []
    st.session_state.vector_store_message = ""
    st.session_state.rag_answer = None
    st.session_state.last_question = ""


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


def display_sidebar(config: AppConfig) -> None:
    """
    Display app status and settings in the sidebar.
    """

    with st.sidebar:
        st.header("📚 AI Study Assistant")

        st.caption("RAG-powered question answering for university notes.")

        st.divider()

        st.subheader("Project status")
        st.success("Phase 1 setup")
        st.success("Phase 2 PDF extraction")
        st.success("Phase 3 text chunking")
        st.success("Phase 4 embeddings")
        st.success("Phase 5 ChromaDB")
        st.success("Phase 6 RAG pipeline")
        st.success("Phase 7 improved UI")
        st.success("Phase 8 citations")

        st.divider()

        if has_api_key(config):
            st.success("API key found")
        else:
            st.warning("API key missing")

        st.subheader("Settings")
        st.write(f"Embedding model: `{config.embedding_model_name}`")
        st.write(f"Chat model: `{config.openai_model}`")
        st.write(f"Chunk size: `{config.chunk_size}`")
        st.write(f"Chunk overlap: `{config.chunk_overlap}`")
        st.write(f"Top K retrieval: `{config.top_k}`")

        threshold_display = (
            "disabled"
            if config.retrieval_distance_threshold is None
            else f"`{config.retrieval_distance_threshold}`"
        )

        st.write(f"Retrieval distance threshold: {threshold_display}")
        st.write(f"Max uploaded PDF size: `{config.max_uploaded_file_mb} MB`")
        st.write(f"ChromaDB directory: `{config.chroma_db_dir}`")
        st.write(f"Collection: `{config.chroma_collection_name}`")

        st.divider()

        if st.button("Clear current session state"):
            clear_working_state()
            st.success("Session state cleared.")


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

    st.caption(
        f"Source: `{selected_page.file_name}`, page `{selected_page.page_number}`"
    )

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
        (
            f"{index + 1}. {chunk.file_name} — "
            f"page {chunk.page_number} — {chunk.chunk_id}"
        )
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
    st.caption(
        f"Source: `{selected_chunk.file_name}`, "
        f"page `{selected_chunk.page_number}`"
    )

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
        (
            f"{index + 1}. {record.file_name} — "
            f"page {record.page_number} — {record.chunk_id}"
        )
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
    st.caption(
        f"Source: `{selected_record.file_name}`, "
        f"page `{selected_record.page_number}`"
    )

    st.text_area(
        "Stored chunk text",
        value=selected_record.text,
        height=220,
        disabled=True,
    )

    return stored_count


def display_rag_answer(rag_answer: RagAnswer | None) -> None:
    """
    Display the generated RAG answer, structured citations, and source chunks.
    """

    if rag_answer is None:
        return

    st.subheader("Answer")
    st.markdown(rag_answer.answer)

    if not rag_answer.sources:
        return

    if rag_answer.citations:
        st.subheader("Citations")

        citations_to_display = get_citations_to_display(
            answer=rag_answer.answer,
            citations=rag_answer.citations,
        )

        for citation in citations_to_display:
            st.markdown(f"- `{format_citation(citation)}`")

    cited_source_numbers = set(extract_source_numbers(rag_answer.answer))

    st.subheader("Retrieved source chunks")

    for index, source in enumerate(rag_answer.sources, start=1):
        source_label = format_source_label(
            index=index,
            file_name=source.file_name,
            page_number=source.page_number,
        )

        if index in cited_source_numbers:
            source_label = f"{source_label} ✅ cited"

        distance_text = (
            f"Similarity distance: `{source.distance:.4f}`"
            if source.distance is not None
            else "Similarity distance: unavailable"
        )

        with st.expander(source_label, expanded=index == 1):
            st.caption(f"Chunk ID: `{source.chunk_id}`")
            st.caption(distance_text)
            st.caption(f"Preview: {truncate_text(source.text, max_chars=180)}")

            st.text_area(
                f"Source {index} text",
                value=source.text,
                height=180,
                disabled=True,
                key=f"rag_source_text_{index}_{source.chunk_id}",
            )


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
            st.write("Step 1/4: Extracting text from PDFs...")
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

            if errors:
                st.write(f"Extraction completed with {len(errors)} warning(s).")

            if not pages:
                status.update(
                    label="Processing stopped: no extractable text found.",
                    state="error",
                )
                st.warning("No selectable text was extracted from the uploaded PDFs.")
                return

            st.write("Step 2/4: Creating text chunks...")
            chunks = chunk_pages(
                pages=pages,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )

            st.session_state.text_chunks = chunks

            if not chunks:
                status.update(
                    label="Processing stopped: no chunks were created.",
                    state="error",
                )
                st.warning("No chunks were created from the extracted text.")
                return

            st.write("Step 3/4: Generating embeddings...")
            embedding_model = get_cached_embedding_model(config.embedding_model_name)

            embedded_chunks = embed_chunks(
                chunks=chunks,
                model=embedding_model,
                batch_size=32,
            )

            st.session_state.embedded_chunks = embedded_chunks

            if not embedded_chunks:
                status.update(
                    label="Processing stopped: no embeddings were generated.",
                    state="error",
                )
                st.warning("No embeddings were generated.")
                return

            st.write("Step 4/4: Storing embeddings in ChromaDB...")
            client = get_cached_chroma_client(str(config.chroma_db_dir))

            if clear_existing_collection:
                reset_collection(
                    client=client,
                    collection_name=config.chroma_collection_name,
                )

            collection = get_or_create_collection(
                client=client,
                collection_name=config.chroma_collection_name,
            )

            upserted_count = upsert_embedded_chunks(
                collection=collection,
                embedded_chunks=embedded_chunks,
            )

            total_count = get_collection_count(collection)

            st.session_state.vector_store_message = (
                f"Stored {upserted_count} chunk(s). "
                f"Collection now contains {total_count} chunk(s)."
            )

            status.update(
                label="Processing complete.",
                state="complete",
            )

            st.success(st.session_state.vector_store_message)

        except (ChunkingError, EmbeddingError, VectorStoreError) as exc:
            status.update(
                label="Processing failed.",
                state="error",
            )
            raise exc


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
                    embedding_model = get_cached_embedding_model(
                        config.embedding_model_name
                    )

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
            "Recommended for the MVP. This prevents notes from previous uploads "
            "being mixed with the current upload."
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


def display_ask_tab(config: AppConfig, stored_count: int) -> None:
    """
    Display the question-answering tab.
    """

    st.header("3. Ask questions")

    st.write(
        """
        Ask a question about the notes currently stored in ChromaDB. The assistant
        will retrieve relevant chunks and answer using only those chunks.
        """
    )

    if stored_count == 0:
        st.info("Process and store notes in ChromaDB before asking questions.")
        return

    if not has_api_key(config):
        st.warning("Add `OPENAI_API_KEY` to your `.env` file before asking questions.")
        return

    if config.retrieval_distance_threshold is not None:
        st.caption(
            f"Retrieval filtering is enabled. Chunks with distance above "
            f"`{config.retrieval_distance_threshold}` will be ignored."
        )
    else:
        st.caption("Retrieval distance filtering is disabled.")

    with st.form("question_form"):
        question = st.text_input(
            "Question",
            placeholder="Example: What is dynamic programming?",
            value=st.session_state.last_question,
        )

        submitted = st.form_submit_button("Ask question", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Retrieving relevant notes and generating answer..."):
                    client = get_cached_chroma_client(str(config.chroma_db_dir))
                    collection = get_or_create_collection(
                        client=client,
                        collection_name=config.chroma_collection_name,
                    )

                    embedding_model = get_cached_embedding_model(
                        config.embedding_model_name
                    )

                    chat_model = get_cached_chat_model(
                        api_key=config.openai_api_key,
                        base_url=config.openai_base_url,
                    )

                    rag_answer = answer_question(
                        question=question,
                        collection=collection,
                        embedding_model=embedding_model,
                        chat_model=chat_model,
                        chat_model_name=config.openai_model,
                        top_k=config.top_k,
                        max_retrieval_distance=config.retrieval_distance_threshold,
                    )

                st.session_state.rag_answer = rag_answer
                st.session_state.last_question = question

            except EmptyQuestionError as exc:
                st.warning(str(exc))

            except ChatModelError as exc:
                st.error(f"Chat model error: {exc}")

            except RagPipelineError as exc:
                st.error(f"RAG pipeline error: {exc}")

            except Exception as exc:
                st.error(f"Unexpected question-answering error: {exc}")

    if st.button("Clear answer"):
        st.session_state.rag_answer = None
        st.session_state.last_question = ""

    display_rag_answer(st.session_state.rag_answer)


def main() -> None:
    st.set_page_config(
        page_title="AI Study Assistant",
        page_icon="📚",
        layout="wide",
    )

    initialise_session_state()

    try:
        config = get_config()
        ensure_storage_dir(config)
    except Exception as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()

    display_sidebar(config)

    st.title("📚 AI Study Assistant for University Notes")

    st.write(
        """
        Upload lecture notes as PDFs, turn them into searchable embeddings, and ask
        questions grounded in your uploaded material.
        """
    )

    process_tab, inspect_tab, ask_tab = st.tabs(
        [
            "1. Process notes",
            "2. Inspect data",
            "3. Ask questions",
        ]
    )

    with process_tab:
        display_process_tab(config)

    with inspect_tab:
        stored_count = display_inspect_tab(config)

    with ask_tab:
        display_ask_tab(config, stored_count)


if __name__ == "__main__":
    main()