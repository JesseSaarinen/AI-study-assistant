import io

import fitz
import pytest

from src.pdf_loader import (
    EmptyPDFError,
    NoSelectableTextError,
    clean_extracted_text,
    extract_text_from_pdf,
)


class FakeUploadedFile(io.BytesIO):
    """
    Small test double that behaves like a Streamlit UploadedFile.
    """

    def __init__(self, name: str, data: bytes) -> None:
        super().__init__(data)
        self.name = name


def create_test_pdf_with_text(text: str) -> bytes:
    """
    Create a small in-memory PDF containing selectable text.
    """

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


def create_blank_test_pdf() -> bytes:
    """
    Create a small in-memory PDF with no text.
    """

    document = fitz.open()
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
