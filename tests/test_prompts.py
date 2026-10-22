from src.models import ChatTurn, RagAnswer, RetrievedChunk
from src.prompts import (
    ANSWER_NOT_FOUND,
    HISTORY_ANSWER_MAX_CHARS,
    REWRITE_SYSTEM_PROMPT,
    build_context_block,
    build_conversation_transcript,
    build_rag_messages,
    build_rewrite_messages,
    strip_citation_markers,
)


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


def make_turn(question: str, answer: str) -> ChatTurn:
    return ChatTurn(question=question, answer=RagAnswer(answer=answer, sources=[]))


def test_strip_citation_markers_removes_inline_and_appended_markers() -> None:
    assert strip_citation_markers("A stack is LIFO. [Source 1][Source 2]") == "A stack is LIFO."
    assert strip_citation_markers("Uses a hash. [Sources 1 and 2] Done.") == "Uses a hash. Done."
    assert strip_citation_markers("A queue is FIFO.\n\nSources: [Source 1], [Source 2]") == (
        "A queue is FIFO."
    )


def test_build_conversation_transcript_shortens_answers_and_drops_markers() -> None:
    long_answer = "Quicksort picks a pivot. " * 50 + "[Source 1]"

    transcript = build_conversation_transcript([make_turn("How does quicksort work?", long_answer)])

    student_line, assistant_line = transcript.splitlines()
    assert student_line == "Student: How does quicksort work?"
    assert assistant_line.startswith("Assistant: Quicksort picks a pivot.")
    assert assistant_line.endswith("...")
    assert len(assistant_line) <= len("Assistant: ") + HISTORY_ANSWER_MAX_CHARS
    assert "[Source" not in transcript


def test_build_rewrite_messages_includes_history_and_latest_question() -> None:
    messages = build_rewrite_messages(
        question="When is it at its worst?",
        history=[make_turn("How does quicksort work?", "It partitions around a pivot.")],
    )

    assert messages[0] == {"role": "system", "content": REWRITE_SYSTEM_PROMPT}
    assert "Student: How does quicksort work?" in messages[1]["content"]
    assert "Assistant: It partitions around a pivot." in messages[1]["content"]
    assert messages[1]["content"].index("How does quicksort work?") < messages[1]["content"].index(
        "When is it at its worst?"
    )
