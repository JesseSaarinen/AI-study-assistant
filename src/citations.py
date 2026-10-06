import re

from src.models import RetrievedChunk, SourceCitation


# Matches bracketed citation groups such as:
# [Source 1]
# [Source 1, Source 2]
# [Source 1, 2]
# [Sources 1 and 2]
#
# The actual number extraction is done separately so that combined markers work.
SOURCE_MARKER_PATTERN = re.compile(
    r"\[\s*Sources?\s+(?P<body>[^\]]+)\]",
    re.IGNORECASE,
)

SOURCE_NUMBER_PATTERN = re.compile(r"\d+")


def build_source_citations(chunks: list[RetrievedChunk]) -> list[SourceCitation]:
    """
    Build structured citations from retrieved chunks.

    The citation source numbers are based on retrieval order:
        first retrieved chunk -> [Source 1]
        second retrieved chunk -> [Source 2]
        etc.
    """

    citations: list[SourceCitation] = []

    for index, chunk in enumerate(chunks, start=1):
        citations.append(
            SourceCitation(
                source_number=index,
                chunk_id=chunk.chunk_id,
                file_name=chunk.file_name,
                page_number=chunk.page_number,
                distance=chunk.distance,
            )
        )

    return citations


def extract_source_numbers(answer: str) -> list[int]:
    """
    Extract source numbers referenced in an answer.

    Supports formats such as:
        [Source 1]
        [Source 1, Source 2]
        [Source 1, 2]
        [Sources 1 and 2]
        [Source 3, Source 4]

    Returns:
        A sorted list of unique source numbers.

    Example:
        "BERT uses bidirectional attention [Source 3, Source 4]."

    Returns:
        [3, 4]
    """

    source_numbers: set[int] = set()

    for marker_match in SOURCE_MARKER_PATTERN.finditer(answer):
        marker_body = marker_match.group("body")

        for number_match in SOURCE_NUMBER_PATTERN.finditer(marker_body):
            source_numbers.add(int(number_match.group()))

    return sorted(source_numbers)


def answer_has_source_markers(answer: str) -> bool:
    """
    Return True if the answer contains at least one source marker.
    """

    return bool(extract_source_numbers(answer))


def select_citations_referenced_in_answer(
    answer: str,
    citations: list[SourceCitation],
) -> list[SourceCitation]:
    """
    Return only citations explicitly referenced in the answer.

    If the answer cites [Source 2], this returns the SourceCitation with
    source_number == 2.

    Combined markers such as [Source 1, Source 2] are also supported.
    """

    referenced_numbers = extract_source_numbers(answer)

    citations_by_number = {
        citation.source_number: citation for citation in citations
    }

    return [
        citations_by_number[source_number]
        for source_number in referenced_numbers
        if source_number in citations_by_number
    ]


def get_citations_to_display(
    answer: str,
    citations: list[SourceCitation],
) -> list[SourceCitation]:
    """
    Decide which citations to display.

    If the answer explicitly references sources, show those.
    If the answer does not reference sources but citations exist, show all
    retrieved citations as supporting context.
    """

    referenced_citations = select_citations_referenced_in_answer(
        answer=answer,
        citations=citations,
    )

    if referenced_citations:
        return referenced_citations

    return citations


def format_citation(citation: SourceCitation) -> str:
    """
    Format a citation as a compact human-readable label.
    """

    page_text = str(citation.page_number) if citation.page_number > 0 else "unknown"

    return f"[Source {citation.source_number}] {citation.file_name}, page {page_text}"


def format_citation_with_distance(citation: SourceCitation) -> str:
    """
    Format a citation with similarity distance if available.
    """

    base = format_citation(citation)

    if citation.distance is None:
        return base

    return f"{base} — distance {citation.distance:.4f}"


def ensure_answer_has_citation_markers(
    answer: str,
    citations: list[SourceCitation],
) -> str:
    """
    Ensure a non-empty answer has citation markers.

    The prompt asks the model to cite using [Source N], but the model may forget.
    This function does not invent new source information. It simply appends the
    retrieved source labels at the end if no citation markers are present.
    """

    cleaned_answer = answer.strip()

    if not cleaned_answer:
        return cleaned_answer

    if not citations:
        return cleaned_answer

    if answer_has_source_markers(cleaned_answer):
        return cleaned_answer

    source_markers = ", ".join(
        f"[Source {citation.source_number}]" for citation in citations
    )

    return f"{cleaned_answer}\n\nSources: {source_markers}"
