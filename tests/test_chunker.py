import pytest

from src.chunker import (
    InvalidChunkingConfigError,
    build_chunk_id,
    chunk_pages,
    normalise_text_for_chunking,
    split_text_into_chunks,
)
from src.models import PageText


def test_normalise_text_for_chunking_removes_empty_lines() -> None:
    raw_text = "  First line  \n\n\nSecond line\u00a0\n"
    cleaned = normalise_text_for_chunking(raw_text)

    assert cleaned == "First line\nSecond line"


def test_short_text_returns_single_chunk() -> None:
    text = "This is a short lecture note."

    chunks = split_text_into_chunks(
        text=text,
        chunk_size=1000,
        chunk_overlap=150,
    )

    assert chunks == [text]


def test_long_text_is_split_into_multiple_chunks() -> None:
    text = "x" * 2500

    chunks = split_text_into_chunks(
        text=text,
        chunk_size=1000,
        chunk_overlap=150,
    )

    assert len(chunks) == 3
    assert all(len(chunk) <= 1000 for chunk in chunks)


def test_chunks_have_overlap() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 100

    chunks = split_text_into_chunks(
        text=text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert chunks[0][-20:] == chunks[1][:20]


def test_invalid_chunk_overlap_raises_error() -> None:
    with pytest.raises(InvalidChunkingConfigError):
        split_text_into_chunks(
            text="Some text",
            chunk_size=100,
            chunk_overlap=100,
        )


def test_negative_chunk_overlap_raises_error() -> None:
    with pytest.raises(InvalidChunkingConfigError):
        split_text_into_chunks(
            text="Some text",
            chunk_size=100,
            chunk_overlap=-1,
        )


def test_build_chunk_id_is_stable() -> None:
    chunk_id_1 = build_chunk_id(
        file_name="Lecture 1.pdf",
        page_number=2,
        chunk_index=1,
        chunk_text="Some chunk text",
    )

    chunk_id_2 = build_chunk_id(
        file_name="Lecture 1.pdf",
        page_number=2,
        chunk_index=1,
        chunk_text="Some chunk text",
    )

    assert chunk_id_1 == chunk_id_2
    assert chunk_id_1.startswith("lecture_1_p2_c1_")


def test_chunk_pages_preserves_metadata() -> None:
    pages = [
        PageText(
            file_name="lecture_1.pdf",
            page_number=3,
            text="This is a lecture note about recursion. " * 100,
        )
    ]

    chunks = chunk_pages(
        pages=pages,
        chunk_size=200,
        chunk_overlap=50,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.file_name == "lecture_1.pdf"
        assert chunk.page_number == 3
        assert chunk.chunk_id.startswith("lecture_1_p3_c")
        assert "recursion" in chunk.text
