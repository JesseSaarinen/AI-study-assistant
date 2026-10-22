from dataclasses import dataclass, field


@dataclass(frozen=True)
class PageText:
    """
    Represents extracted text from a single PDF page.
    """

    file_name: str
    page_number: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    """
    Represents a chunk of text created from extracted PDF pages.
    """

    chunk_id: str
    file_name: str
    page_number: int
    text: str


@dataclass(frozen=True)
class EmbeddedChunk:
    """
    Represents a text chunk together with its embedding vector.
    """

    chunk: TextChunk
    embedding: list[float]


@dataclass(frozen=True)
class RetrievedChunk:
    """
    Represents a chunk retrieved from the vector database.
    """

    chunk_id: str
    file_name: str
    page_number: int
    text: str
    distance: float | None = None


@dataclass(frozen=True)
class SourceCitation:
    """
    Represents a source citation attached to a retrieved chunk.

    source_number corresponds to labels like [Source 1], [Source 2], etc.
    """

    source_number: int
    chunk_id: str
    file_name: str
    page_number: int
    distance: float | None = None


@dataclass(frozen=True)
class RagAnswer:
    """
    Represents the final RAG answer returned to the UI.

    search_query is the question that was actually used for retrieval. It
    differs from the user's question when a follow-up was rewritten.
    """

    answer: str
    sources: list[RetrievedChunk]
    citations: list[SourceCitation] = field(default_factory=list)
    search_query: str = ""


@dataclass(frozen=True)
class ChatTurn:
    """
    One question and answer in a conversation.

    Earlier turns let the pipeline rewrite follow-up questions such as
    "why is that?" into standalone questions before retrieval.
    """

    question: str
    answer: RagAnswer
