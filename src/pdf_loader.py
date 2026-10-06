from pathlib import Path
from typing import Iterable, Protocol

import fitz

from src.models import PageText


class UploadedPDFLike(Protocol):
    """
    Minimal file-like interface needed for uploaded PDFs.

    Streamlit's UploadedFile matches this shape, but we avoid importing Streamlit
    here so this module stays reusable and easy to test.
    """

    name: str

    def read(self) -> bytes:
        ...

    def seek(self, offset: int, whence: int = 0) -> int:
        ...


class PDFExtractionError(Exception):
    """
    Base exception for PDF extraction errors.
    """


class EmptyPDFError(PDFExtractionError):
    """
    Raised when an uploaded PDF is empty or has no pages.
    """


class InvalidPDFTypeError(PDFExtractionError):
    """
    Raised when the uploaded file does not look like a PDF.
    """


class FileTooLargeError(PDFExtractionError):
    """
    Raised when an uploaded PDF exceeds the configured size limit.
    """


class InvalidPDFError(PDFExtractionError):
    """
    Raised when a file cannot be opened as a valid PDF.
    """


class NoSelectableTextError(PDFExtractionError):
    """
    Raised when a PDF has pages but no selectable text.

    This often happens with scanned/image-only PDFs.
    """


def _safe_file_name(uploaded_file: UploadedPDFLike) -> str:
    """
    Return a safe display name for an uploaded file.
    """

    raw_name = getattr(uploaded_file, "name", "uploaded.pdf")
    file_name = Path(raw_name).name.strip()

    return file_name or "uploaded.pdf"


def _validate_pdf_file_name(file_name: str) -> None:
    """
    Validate that the uploaded file has a PDF extension.

    Streamlit already filters by extension in the UI, but this makes the backend
    safer and easier to test.
    """

    if not file_name.lower().endswith(".pdf"):
        raise InvalidPDFTypeError(
            f"{file_name}: invalid file type. Please upload a PDF file."
        )


def _read_uploaded_file(uploaded_file: UploadedPDFLike, file_name: str) -> bytes:
    """
    Read all bytes from an uploaded file.

    We try to reset the pointer first so repeated reads work reliably.
    """

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    try:
        data = uploaded_file.read()
    except Exception as exc:
        raise PDFExtractionError(f"{file_name}: failed to read uploaded file.") from exc

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    return data


def _validate_pdf_bytes(
    pdf_bytes: bytes,
    file_name: str,
    max_file_size_mb: int | None = None,
) -> None:
    """
    Validate uploaded PDF bytes before passing them to PyMuPDF.
    """

    if not pdf_bytes:
        raise EmptyPDFError(f"{file_name}: the uploaded file is empty.")

    if max_file_size_mb is not None:
        if max_file_size_mb <= 0:
            raise PDFExtractionError("max_file_size_mb must be greater than 0.")

        max_bytes = max_file_size_mb * 1024 * 1024

        if len(pdf_bytes) > max_bytes:
            actual_size_mb = len(pdf_bytes) / (1024 * 1024)

            raise FileTooLargeError(
                f"{file_name}: file is too large "
                f"({actual_size_mb:.1f} MB). Maximum allowed size is "
                f"{max_file_size_mb} MB."
            )

    # Most PDFs contain the %PDF marker near the start of the file.
    # We check the first 1024 bytes rather than only the first bytes because
    # some PDFs may include a small preamble.
    if b"%PDF" not in pdf_bytes[:1024]:
        raise InvalidPDFTypeError(
            f"{file_name}: this file does not appear to be a valid PDF."
        )


def clean_extracted_text(text: str) -> str:
    """
    Clean raw PDF text while preserving useful line breaks.

    PDF extraction can include odd whitespace, null characters, and empty lines.
    This function normalises the most common issues without being too aggressive.
    """

    text = text.replace("\x00", " ")
    text = text.replace("\u00a0", " ")

    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]

    return "\n".join(non_empty_lines).strip()


def extract_text_from_pdf(
    uploaded_file: UploadedPDFLike,
    max_file_size_mb: int | None = None,
) -> list[PageText]:
    """
    Extract selectable text from a single uploaded PDF, page by page.

    Args:
        uploaded_file: A Streamlit UploadedFile or any similar file-like object.
        max_file_size_mb: Optional maximum allowed PDF file size.

    Returns:
        A list of PageText objects containing file name, page number, and text.

    Raises:
        EmptyPDFError: If the file is empty or the PDF contains no pages.
        InvalidPDFTypeError: If the file does not look like a PDF.
        FileTooLargeError: If the file exceeds the configured size limit.
        InvalidPDFError: If the file cannot be opened as a PDF.
        NoSelectableTextError: If no selectable text is found.
        PDFExtractionError: For other extraction failures.
    """

    file_name = _safe_file_name(uploaded_file)
    _validate_pdf_file_name(file_name)

    pdf_bytes = _read_uploaded_file(uploaded_file, file_name)
    _validate_pdf_bytes(
        pdf_bytes=pdf_bytes,
        file_name=file_name,
        max_file_size_mb=max_file_size_mb,
    )

    document = None

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise InvalidPDFError(
            f"{file_name}: could not be opened as a PDF. "
            "The file may be corrupted or not a valid PDF."
        ) from exc

    try:
        if document.page_count == 0:
            raise EmptyPDFError(f"{file_name}: the PDF contains no pages.")

        extracted_pages: list[PageText] = []

        for page_index in range(document.page_count):
            page_number = page_index + 1

            try:
                page = document.load_page(page_index)
                raw_text = page.get_text("text")
                cleaned_text = clean_extracted_text(raw_text)
            except Exception as exc:
                raise PDFExtractionError(
                    f"{file_name}: failed to extract text from page {page_number}."
                ) from exc

            if cleaned_text:
                extracted_pages.append(
                    PageText(
                        file_name=file_name,
                        page_number=page_number,
                        text=cleaned_text,
                    )
                )

        if not extracted_pages:
            raise NoSelectableTextError(
                f"{file_name}: no selectable text was found. "
                "This may be a scanned/image-only PDF. OCR would be needed for this file."
            )

        return extracted_pages

    finally:
        if document is not None:
            document.close()


def extract_text_from_pdfs(
    uploaded_files: Iterable[UploadedPDFLike],
    max_file_size_mb: int | None = None,
) -> tuple[list[PageText], list[str]]:
    """
    Extract text from multiple uploaded PDFs.

    This function does not stop the whole process if one PDF fails.
    Instead, it returns successfully extracted pages and a list of error messages.

    Args:
        uploaded_files: Iterable of uploaded PDF-like files.
        max_file_size_mb: Optional maximum allowed size for each uploaded PDF.

    Returns:
        A tuple:
            - list of extracted PageText objects
            - list of error messages
    """

    all_pages: list[PageText] = []
    errors: list[str] = []

    for uploaded_file in uploaded_files:
        try:
            pages = extract_text_from_pdf(
                uploaded_file=uploaded_file,
                max_file_size_mb=max_file_size_mb,
            )
            all_pages.extend(pages)
        except PDFExtractionError as exc:
            errors.append(str(exc))

    return all_pages, errors
