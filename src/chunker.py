import hashlib
import re
from pathlib import Path

from src.models import PageText, TextChunk


class ChunkingError(Exception):
    """
    Base exception for text chunking errors.
    """


class InvalidChunkingConfigError(ChunkingError):
    """
    Raised when chunking settings are invalid.
    """


def validate_chunking_params(chunk_size: int, chunk_overlap: int) -> None:
    """
    Validate chunk size and overlap settings.

    Args:
        chunk_size: Maximum number of characters in each chunk.
        chunk_overlap: Number of characters repeated between neighbouring chunks.

    Raises:
        InvalidChunkingConfigError: If the settings are invalid.
    """

    if chunk_size <= 0:
        raise InvalidChunkingConfigError("CHUNK_SIZE must be greater than 0.")

    if chunk_overlap < 0:
        raise InvalidChunkingConfigError("CHUNK_OVERLAP cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise InvalidChunkingConfigError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )


def normalise_text_for_chunking(text: str) -> str:
    """
    Normalise text before chunking.

    This keeps useful line breaks but removes common PDF extraction artefacts,
    such as null characters, non-breaking spaces, and repeated blank lines.
    """

    text = text.replace("\x00", " ")
    text = text.replace("\u00a0", " ")

    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]

    return "\n".join(non_empty_lines).strip()


def _find_best_split_position(text: str, start: int, target_end: int) -> int:
    """
    Find a natural split point close to target_end.

    Instead of always splitting exactly at chunk_size, this tries to split near
    a paragraph break, line break, sentence boundary, comma, or space.

    If no good boundary is found, it falls back to target_end.
    """

    if target_end >= len(text):
        return len(text)

    current_chunk_length = target_end - start
    minimum_split_position = start + max(1, int(current_chunk_length * 0.5))

    delimiters = [
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ", ",
        " ",
    ]

    best_position = -1
    best_delimiter_length = 0

    for delimiter in delimiters:
        position = text.rfind(delimiter, minimum_split_position, target_end)

        if position > best_position:
            best_position = position
            best_delimiter_length = len(delimiter)

    if best_position == -1:
        return target_end

    return best_position + best_delimiter_length


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: The text to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters repeated between consecutive chunks.

    Returns:
        A list of chunk strings.
    """

    validate_chunking_params(chunk_size, chunk_overlap)

    cleaned_text = normalise_text_for_chunking(text)

    if not cleaned_text:
        return []

    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        target_end = min(start + chunk_size, text_length)

        if target_end < text_length:
            end = _find_best_split_position(cleaned_text, start, target_end)
        else:
            end = text_length

        if end <= start:
            end = target_end

        chunk = cleaned_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def _slugify_file_stem(file_name: str) -> str:
    """
    Convert a file name into a safe ID component.
    """

    stem = Path(file_name).stem
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem)
    slug = slug.strip("_").lower()

    return slug or "document"


def build_chunk_id(
    file_name: str,
    page_number: int,
    chunk_index: int,
    chunk_text: str,
) -> str:
    """
    Build a stable unique chunk ID.

    The hash helps avoid collisions if different files/pages have similar names.
    """

    slug = _slugify_file_stem(file_name)

    hash_input = f"{file_name}:{page_number}:{chunk_index}:{chunk_text}"
    digest = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:12]

    return f"{slug}_p{page_number}_c{chunk_index}_{digest}"


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[TextChunk]:
    """
    Convert extracted PDF pages into metadata-preserving text chunks.

    Each page is chunked independently so that every chunk keeps a clear page
    number for source citation.

    Args:
        pages: Extracted PDF pages.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters.

    Returns:
        A list of TextChunk objects.
    """

    validate_chunking_params(chunk_size, chunk_overlap)

    all_chunks: list[TextChunk] = []

    for page in pages:
        page_chunks = split_text_into_chunks(
            text=page.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, chunk_text in enumerate(page_chunks, start=1):
            chunk_id = build_chunk_id(
                file_name=page.file_name,
                page_number=page.page_number,
                chunk_index=chunk_index,
                chunk_text=chunk_text,
            )

            all_chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    file_name=page.file_name,
                    page_number=page.page_number,
                    text=chunk_text,
                )
            )

    return all_chunks
