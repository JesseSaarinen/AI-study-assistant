import pytest

from src.citations import (
    answer_has_source_markers,
    build_source_citations,
    ensure_answer_has_citation_markers,
    extract_source_numbers,
    format_citation,
    format_citation_with_distance,
    get_citations_to_display,
    select_citations_referenced_in_answer,
)
from src.models import RetrievedChunk


def make_retrieved_chunk(
    chunk_id: str,
    file_name: str = "lecture.pdf",
    page_number: int = 1,
    text: str = "Example source text.",
    distance: float | None = 0.1234,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        file_name=file_name,
        page_number=page_number,
        text=text,
        distance=distance,
    )


def test_build_source_citations_assigns_source_numbers() -> None:
    chunks = [
        make_retrieved_chunk("chunk_1", file_name="a.pdf", page_number=2),
        make_retrieved_chunk("chunk_2", file_name="b.pdf", page_number=5),
    ]

    citations = build_source_citations(chunks)

    assert len(citations) == 2
    assert citations[0].source_number == 1
    assert citations[0].chunk_id == "chunk_1"
    assert citations[0].file_name == "a.pdf"
    assert citations[0].page_number == 2

    assert citations[1].source_number == 2
    assert citations[1].chunk_id == "chunk_2"
    assert citations[1].file_name == "b.pdf"
    assert citations[1].page_number == 5


@pytest.mark.parametrize(
    ("answer", "expected_source_numbers"),
    [
        (
            "Recursion is when a function calls itself. [Source 1]",
            [1],
        ),
        (
            "BERT uses bidirectional attention. [Source 1, Source 2]",
            [1, 2],
        ),
        (
            "BERT uses bidirectional attention. [Source 1, 2]",
            [1, 2],
        ),
        (
            "BERT and GPT use different Transformer architectures. [Sources 1 and 2]",
            [1, 2],
        ),
        (
            "BERT is encoder-only and GPT is decoder-only. [Source 3, Source 4]",
            [3, 4],
        ),
        (
            "Multiple adjacent markers are also valid. [Source 1][Source 2]",
            [1, 2],
        ),
        (
            "Lowercase markers should also work. [source 2, source 4]",
            [2, 4],
        ),
    ],
)
def test_extract_source_numbers_supports_common_marker_formats(
    answer: str,
    expected_source_numbers: list[int],
) -> None:
    assert extract_source_numbers(answer) == expected_source_numbers


def test_extract_source_numbers_returns_unique_sorted_numbers() -> None:
    answer = (
        "First claim [Source 3]. "
        "Second claim [Source 1]. "
        "Repeated source [Source 3]. "
        "Combined source marker [Sources 2 and 1]."
    )

    source_numbers = extract_source_numbers(answer)

    assert source_numbers == [1, 2, 3]


def test_extract_source_numbers_returns_empty_list_when_no_markers() -> None:
    answer = "This answer has no source markers."

    source_numbers = extract_source_numbers(answer)

    assert source_numbers == []


def test_answer_has_source_markers() -> None:
    assert answer_has_source_markers("This is supported. [Source 1]")
    assert answer_has_source_markers("This is supported. [Source 1, Source 2]")
    assert answer_has_source_markers("This is supported. [Sources 1 and 2]")
    assert not answer_has_source_markers("This answer has no markers.")


def test_select_citations_referenced_in_answer() -> None:
    chunks = [
        make_retrieved_chunk("chunk_1"),
        make_retrieved_chunk("chunk_2"),
        make_retrieved_chunk("chunk_3"),
    ]
    citations = build_source_citations(chunks)

    selected = select_citations_referenced_in_answer(
        answer="This uses [Source 3] and [Source 1].",
        citations=citations,
    )

    assert [citation.source_number for citation in selected] == [1, 3]


def test_select_citations_referenced_in_answer_handles_combined_markers() -> None:
    chunks = [
        make_retrieved_chunk("chunk_1"),
        make_retrieved_chunk("chunk_2"),
        make_retrieved_chunk("chunk_3"),
        make_retrieved_chunk("chunk_4"),
    ]
    citations = build_source_citations(chunks)

    selected = select_citations_referenced_in_answer(
        answer="This uses combined markers. [Source 3, Source 4]",
        citations=citations,
    )

    assert [citation.source_number for citation in selected] == [3, 4]


def test_get_citations_to_display_returns_referenced_citations_when_present() -> None:
    chunks = [
        make_retrieved_chunk("chunk_1"),
        make_retrieved_chunk("chunk_2"),
    ]
    citations = build_source_citations(chunks)

    displayed = get_citations_to_display(
        answer="This uses only [Source 2].",
        citations=citations,
    )

    assert len(displayed) == 1
    assert displayed[0].source_number == 2


def test_get_citations_to_display_handles_combined_markers() -> None:
    chunks = [
        make_retrieved_chunk("chunk_1"),
        make_retrieved_chunk("chunk_2"),
        make_retrieved_chunk("chunk_3"),
        make_retrieved_chunk("chunk_4"),
    ]
    citations = build_source_citations(chunks)

    displayed = get_citations_to_display(
        answer="This answer uses multiple sources. [Source 1, Source 4]",
        citations=citations,
    )

    assert [citation.source_number for citation in displayed] == [1, 4]


def test_get_citations_to_display_returns_all_when_answer_has_no_markers() -> None:
    chunks = [
        make_retrieved_chunk("chunk_1"),
        make_retrieved_chunk("chunk_2"),
    ]
    citations = build_source_citations(chunks)

    displayed = get_citations_to_display(
        answer="This answer forgot source markers.",
        citations=citations,
    )

    assert displayed == citations


def test_format_citation() -> None:
    citation = build_source_citations(
        [
            make_retrieved_chunk(
                "chunk_1",
                file_name="algorithms.pdf",
                page_number=7,
            )
        ]
    )[0]

    assert format_citation(citation) == "[Source 1] algorithms.pdf, page 7"


def test_format_citation_with_distance() -> None:
    citation = build_source_citations(
        [
            make_retrieved_chunk(
                "chunk_1",
                file_name="algorithms.pdf",
                page_number=7,
                distance=0.123456,
            )
        ]
    )[0]

    assert (
        format_citation_with_distance(citation)
        == "[Source 1] algorithms.pdf, page 7 — distance 0.1235"
    )


def test_ensure_answer_has_citation_markers_appends_sources_when_missing() -> None:
    chunks = [
        make_retrieved_chunk("chunk_1"),
        make_retrieved_chunk("chunk_2"),
    ]
    citations = build_source_citations(chunks)

    answer = ensure_answer_has_citation_markers(
        answer="Recursion is when a function calls itself.",
        citations=citations,
    )

    assert "Recursion is when a function calls itself." in answer
    assert "Sources: [Source 1], [Source 2]" in answer


def test_ensure_answer_has_citation_markers_does_not_duplicate_existing_marker() -> None:
    chunks = [
        make_retrieved_chunk("chunk_1"),
    ]
    citations = build_source_citations(chunks)

    answer = ensure_answer_has_citation_markers(
        answer="Recursion is when a function calls itself. [Source 1]",
        citations=citations,
    )

    assert answer == "Recursion is when a function calls itself. [Source 1]"


def test_ensure_answer_has_citation_markers_does_not_duplicate_combined_markers() -> None:
    chunks = [
        make_retrieved_chunk("chunk_1"),
        make_retrieved_chunk("chunk_2"),
    ]
    citations = build_source_citations(chunks)

    original_answer = "BERT uses bidirectional attention. [Source 1, Source 2]"

    answer = ensure_answer_has_citation_markers(
        answer=original_answer,
        citations=citations,
    )

    assert answer == original_answer
