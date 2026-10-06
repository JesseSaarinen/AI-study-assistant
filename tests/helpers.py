import io

import fitz

from src.models import EmbeddedChunk, TextChunk


class FakeUploadedFile(io.BytesIO):
    """
    Test double that behaves like a Streamlit UploadedFile.
    """

    def __init__(self, name: str, data: bytes) -> None:
        super().__init__(data)
        self.name = name


def create_pdf_with_pages(page_texts: list[str]) -> bytes:
    """
    Create an in-memory PDF with one page per text item.

    Args:
        page_texts: List of strings. Each string becomes one PDF page.

    Returns:
        PDF bytes.
    """

    document = fitz.open()

    for text in page_texts:
        page = document.new_page()

        if text:
            page.insert_text(
                point=(72, 72),
                text=text,
                fontsize=11,
            )

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


class KeywordFakeEmbeddingModel:
    """
    Fake embedding model for deterministic tests.

    It maps:
    - recursion-related text to [1, 0]
    - database/SQL-related text to [0, 1]
    - other text to [0.5, 0.5]

    This allows semantic retrieval tests without downloading a real model.
    """

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> list[list[float]]:
        embeddings: list[list[float]] = []

        for sentence in sentences:
            lowered = sentence.lower()

            if "database" in lowered or "sql" in lowered or "relational" in lowered:
                embeddings.append([0.0, 1.0])
            elif "recursion" in lowered or "recursive" in lowered or "base case" in lowered:
                embeddings.append([1.0, 0.0])
            else:
                embeddings.append([0.5, 0.5])

        return embeddings


class FixedAnswerChatModel:
    """
    Fake chat model that returns a fixed answer and records whether it was called.
    """

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.was_called = False
        self.last_messages: list[dict[str, str]] = []

    def generate_answer(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
    ) -> str:
        self.was_called = True
        self.last_messages = messages
        return self.answer


def make_embedded_chunk(
    chunk_id: str,
    text: str,
    embedding: list[float],
    file_name: str = "lecture.pdf",
    page_number: int = 1,
) -> EmbeddedChunk:
    """
    Create an EmbeddedChunk for tests.
    """

    return EmbeddedChunk(
        chunk=TextChunk(
            chunk_id=chunk_id,
            file_name=file_name,
            page_number=page_number,
            text=text,
        ),
        embedding=embedding,
    )
