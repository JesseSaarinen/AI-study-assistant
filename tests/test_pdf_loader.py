import pymupdf
import pytest

from src.pdf_loader import (
    EmptyPDFError,
    FileTooLargeError,
    InvalidPDFTypeError,
    NoSelectableTextError,
    clean_extracted_text,
    extract_text_from_pdf,
    extract_text_from_pdfs,
)
from tests.helpers import FakeUploadedFile, create_pdf_with_pages


def create_test_pdf_with_text(text: str) -> bytes:
    """
    Create a small in-memory PDF containing selectable text.
    """

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


def create_blank_test_pdf() -> bytes:
    """
    Create a small in-memory PDF with no text.
    """

    document = pymupdf.open()
    document.new_page()

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


def test_clean_extracted_text_removes_empty_lines() -> None:
    raw_text = "  Line one  \n\n\n  Line two\u00a0 \n"
    cleaned = clean_extracted_text(raw_text)

    assert cleaned == "Line one\nLine two"


def test_extract_text_from_pdf_returns_page_text() -> None:
    pdf_bytes = create_test_pdf_with_text("This is a test lecture note.")
    uploaded_file = FakeUploadedFile("lecture_test.pdf", pdf_bytes)

    pages = extract_text_from_pdf(uploaded_file)

    assert len(pages) == 1
    assert pages[0].file_name == "lecture_test.pdf"
    assert pages[0].page_number == 1
    assert "test lecture note" in pages[0].text


def test_extract_text_from_empty_file_raises_error() -> None:
    uploaded_file = FakeUploadedFile("empty.pdf", b"")

    with pytest.raises(EmptyPDFError):
        extract_text_from_pdf(uploaded_file)


def test_extract_text_from_blank_pdf_raises_no_selectable_text_error() -> None:
    pdf_bytes = create_blank_test_pdf()
    uploaded_file = FakeUploadedFile("blank.pdf", pdf_bytes)

    with pytest.raises(NoSelectableTextError):
        extract_text_from_pdf(uploaded_file)


def test_extract_text_rejects_non_pdf_extension() -> None:
    uploaded_file = FakeUploadedFile("notes.txt", b"This is not a PDF.")

    with pytest.raises(InvalidPDFTypeError):
        extract_text_from_pdf(uploaded_file)


def test_extract_text_rejects_invalid_pdf_bytes() -> None:
    uploaded_file = FakeUploadedFile("notes.pdf", b"This is not really a PDF.")

    with pytest.raises(InvalidPDFTypeError):
        extract_text_from_pdf(uploaded_file)


def test_extract_text_rejects_large_file_when_limit_exceeded() -> None:
    pdf_bytes = create_test_pdf_with_text("This is a test PDF.")
    uploaded_file = FakeUploadedFile("large.pdf", pdf_bytes)

    with pytest.raises(FileTooLargeError):
        extract_text_from_pdf(
            uploaded_file=uploaded_file,
            max_file_size_mb=0.0001,
        )


def test_extract_text_from_pdfs_continues_after_invalid_file() -> None:
    """
    One bad file should not stop all PDFs from being processed.
    """

    valid_pdf = FakeUploadedFile(
        "valid_notes.pdf",
        create_pdf_with_pages(["This page contains selectable lecture note text."]),
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
