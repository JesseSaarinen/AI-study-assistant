import pytest

from src.ui_helpers import (
    build_pipeline_status_items,
    format_count,
    format_source_label,
    status_icon,
    truncate_text,
)


def test_format_count_uses_singular_for_one() -> None:
    assert format_count(1, "page") == "1 page"


def test_format_count_uses_plural_for_multiple() -> None:
    assert format_count(3, "chunk") == "3 chunks"


def test_format_count_supports_custom_plural() -> None:
    assert format_count(2, "entry", "entries") == "2 entries"


def test_truncate_text_returns_short_text_unchanged() -> None:
    assert truncate_text("This is short.", max_chars=50) == "This is short."


def test_truncate_text_shortens_long_text() -> None:
    text = "word " * 100

    truncated = truncate_text(text, max_chars=30)

    assert len(truncated) <= 30
    assert truncated.endswith("...")


def test_truncate_text_rejects_tiny_limit() -> None:
    with pytest.raises(ValueError):
        truncate_text("Some text", max_chars=3)


def test_format_source_label() -> None:
    label = format_source_label(
        index=2,
        file_name="lecture.pdf",
        page_number=7,
    )

    assert label == "[Source 2] lecture.pdf, page 7"


def test_status_icon() -> None:
    assert status_icon(True) == "✅"
    assert status_icon(False) == "⬜"


def test_build_pipeline_status_items() -> None:
    items = build_pipeline_status_items(
        page_count=2,
        chunk_count=5,
        embedded_count=5,
        stored_count=0,
    )

    assert len(items) == 4
    assert "✅ **PDF extraction**" in items[0]
    assert "✅ **Text chunking**" in items[1]
    assert "✅ **Embedding generation**" in items[2]
    assert "⬜ **Vector storage**" in items[3]
