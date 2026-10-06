import io

import fitz
import pytest

from src.pdf_loader import (
    FileTooLargeError,
    InvalidPDFTypeError,
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
