from src.models import RetrievedChunk
from src.prompts import ANSWER_NOT_FOUND, build_context_block, build_rag_messages


def test_build_context_block_includes_source_metadata() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk_1",
            file_name="lecture.pdf",
            page_number=3,
            text="Recursion is when a function calls itself.",
            distance=0.1,
        )
    ]

    context = build_context_block(chunks)

    assert "[Source 1]" in context
    assert "File: lecture.pdf" in context
    assert "Page: 3" in context
    assert "function calls itself" in context


def test_build_rag_messages_includes_question_and_fallback_instruction() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk_1",
            file_name="lecture.pdf",
            page_number=1,
            text="A stack is a last-in, first-out data structure.",
            distance=0.2,
        )
    ]

    messages = build_rag_messages(
        question="What is a stack?",
        chunks=chunks,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "What is a stack?" in messages[1]["content"]
    assert ANSWER_NOT_FOUND in messages[1]["content"]
    assert "A stack is a last-in, first-out" in messages[1]["content"]
