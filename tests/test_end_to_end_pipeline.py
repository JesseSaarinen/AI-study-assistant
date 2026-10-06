from src.chunker import chunk_pages
from src.embeddings import embed_chunks
from src.prompts import ANSWER_NOT_FOUND
from src.pdf_loader import extract_text_from_pdf
from src.rag_pipeline import answer_question
from src.vector_store import (
    get_chroma_client,
    get_collection_count,
    get_or_create_collection,
    upsert_embedded_chunks,
)
from tests.helpers import (
    FakeUploadedFile,
    FixedAnswerChatModel,
    KeywordFakeEmbeddingModel,
    create_pdf_with_pages,
)


def test_full_local_rag_pipeline_without_external_api(tmp_path) -> None:
    """
    Test the core RAG pipeline without Streamlit or OpenAI.

    This verifies that the main project flow works end to end:
    PDF -> pages -> chunks -> embeddings -> ChromaDB -> answer.
    """

    pdf_bytes = create_pdf_with_pages(
        [
            (
                "Recursion is when a function calls itself. "
                "A base case stops the recursive calls."
            )
        ]
    )

    uploaded_file = FakeUploadedFile("algorithms.pdf", pdf_bytes)

    pages = extract_text_from_pdf(
        uploaded_file=uploaded_file,
        max_file_size_mb=5,
    )

    assert len(pages) == 1
    assert pages[0].file_name == "algorithms.pdf"
    assert pages[0].page_number == 1
    assert "Recursion" in pages[0].text

    chunks = chunk_pages(
        pages=pages,
        chunk_size=180,
        chunk_overlap=30,
    )

    assert len(chunks) >= 1
    assert chunks[0].file_name == "algorithms.pdf"
    assert chunks[0].page_number == 1

    embedding_model = KeywordFakeEmbeddingModel()

    embedded_chunks = embed_chunks(
        chunks=chunks,
        model=embedding_model,
        batch_size=2,
    )

    assert len(embedded_chunks) == len(chunks)

    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "phase10_notes")

    upserted_count = upsert_embedded_chunks(
        collection=collection,
        embedded_chunks=embedded_chunks,
    )

    assert upserted_count == len(embedded_chunks)
    assert get_collection_count(collection) == len(embedded_chunks)

    chat_model = FixedAnswerChatModel(
        "Recursion is when a function calls itself. [Source 1]"
    )

    rag_answer = answer_question(
        question="What is recursion?",
        collection=collection,
        embedding_model=embedding_model,
        chat_model=chat_model,
        chat_model_name="fake-model",
        top_k=2,
        max_retrieval_distance=0.25,
    )

    assert chat_model.was_called is True
    assert "function calls itself" in rag_answer.answer
    assert "[Source 1]" in rag_answer.answer
    assert len(rag_answer.sources) >= 1
    assert len(rag_answer.citations) >= 1
    assert rag_answer.sources[0].file_name == "algorithms.pdf"
    assert rag_answer.sources[0].page_number == 1


def test_full_local_rag_pipeline_returns_fallback_when_threshold_filters_context(
    tmp_path,
) -> None:
    """
    Test that weak retrieval results are filtered before calling the chat model.
    """

    pdf_bytes = create_pdf_with_pages(
        [
            "Recursion is when a function calls itself."
        ]
    )

    uploaded_file = FakeUploadedFile("algorithms.pdf", pdf_bytes)

    pages = extract_text_from_pdf(
        uploaded_file=uploaded_file,
        max_file_size_mb=5,
    )

    chunks = chunk_pages(
        pages=pages,
        chunk_size=180,
        chunk_overlap=30,
    )

    embedding_model = KeywordFakeEmbeddingModel()
    embedded_chunks = embed_chunks(chunks=chunks, model=embedding_model)

    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "phase10_threshold_notes")

    upsert_embedded_chunks(
        collection=collection,
        embedded_chunks=embedded_chunks,
    )

    chat_model = FixedAnswerChatModel("This should not be called.")

    rag_answer = answer_question(
        question="What is SQL?",
        collection=collection,
        embedding_model=embedding_model,
        chat_model=chat_model,
        chat_model_name="fake-model",
        top_k=1,
        max_retrieval_distance=0.5,
    )

    assert rag_answer.answer == ANSWER_NOT_FOUND
    assert rag_answer.sources == []
    assert rag_answer.citations == []
    assert chat_model.was_called is False


def test_rag_pipeline_appends_citations_when_chat_model_forgets_markers(
    tmp_path,
) -> None:
    """
    Test the Phase 8 citation fallback inside the full RAG pipeline.
    """

    pdf_bytes = create_pdf_with_pages(
        [
            "Recursion is when a function calls itself."
        ]
    )

    uploaded_file = FakeUploadedFile("algorithms.pdf", pdf_bytes)

    pages = extract_text_from_pdf(
        uploaded_file=uploaded_file,
        max_file_size_mb=5,
    )

    chunks = chunk_pages(
        pages=pages,
        chunk_size=180,
        chunk_overlap=30,
    )

    embedding_model = KeywordFakeEmbeddingModel()
    embedded_chunks = embed_chunks(chunks=chunks, model=embedding_model)

    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "phase10_citation_notes")

    upsert_embedded_chunks(
        collection=collection,
        embedded_chunks=embedded_chunks,
    )

    chat_model = FixedAnswerChatModel(
        "Recursion is when a function calls itself."
    )

    rag_answer = answer_question(
        question="What is recursion?",
        collection=collection,
        embedding_model=embedding_model,
        chat_model=chat_model,
        chat_model_name="fake-model",
        top_k=1,
        max_retrieval_distance=0.25,
    )

    assert chat_model.was_called is True
    assert "Recursion is when a function calls itself." in rag_answer.answer
    assert "Sources: [Source 1]" in rag_answer.answer
    assert len(rag_answer.citations) == 1
