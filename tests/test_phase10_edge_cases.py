import pytest

from src.chunker import chunk_pages
from src.embeddings import EmbeddingGenerationError, generate_embeddings
from src.models import PageText
from src.pdf_loader import extract_text_from_pdfs
from src.vector_store import (
    InvalidCollectionNameError,
    validate_collection_name,
)
from tests.helpers import FakeUploadedFile, create_pdf_with_pages


class InvalidVectorOutputModel:
    """
    Fake model that returns invalid vector output.
    """

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> list[str]:
        return ["not-a-vector" for _ in sentences]


class NonNumericVectorOutputModel:
    """
    Fake model that returns a vector containing a non-numeric value.
    """

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> list[list[object]]:
        return [[1.0, "not-a-number"] for _ in sentences]


def test_extract_text_from_pdfs_continues_after_invalid_file() -> None:
    """
    One bad file should not stop all PDFs from being processed.
    """

    valid_pdf = FakeUploadedFile(
        "valid_notes.pdf",
        create_pdf_with_pages(
            [
                "This page contains selectable lecture note text."
            ]
        ),
    )

    invalid_file = FakeUploadedFile(
        "not_a_pdf.txt",
        b"This is not a PDF file.",
    )

    pages, errors = extract_text_from_pdfs(
        uploaded_files=[valid_pdf, invalid_file],
        max_file_size_mb=5,
    )

    assert len(pages) == 1
    assert pages[0].file_name == "valid_notes.pdf"
    assert "selectable lecture note text" in pages[0].text

    assert len(errors) == 1
    assert "not_a_pdf.txt" in errors[0]


def test_chunk_ids_are_unique_across_pages_and_files() -> None:
    pages = [
        PageText(
            file_name="lecture.pdf",
            page_number=1,
            text="Same text about algorithms.",
        ),
        PageText(
            file_name="lecture.pdf",
            page_number=2,
            text="Same text about algorithms.",
        ),
        PageText(
            file_name="other_lecture.pdf",
            page_number=1,
            text="Same text about algorithms.",
        ),
    ]

    chunks = chunk_pages(
        pages=pages,
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunks) == 3
    assert len(set(chunk_ids)) == 3


def test_generate_embeddings_rejects_invalid_vector_output() -> None:
    model = InvalidVectorOutputModel()

    with pytest.raises(EmbeddingGenerationError):
        generate_embeddings(
            texts=["Some valid text."],
            model=model,
        )


def test_generate_embeddings_rejects_non_numeric_vector_values() -> None:
    model = NonNumericVectorOutputModel()

    with pytest.raises(EmbeddingGenerationError):
        generate_embeddings(
            texts=["Some valid text."],
            model=model,
        )


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
