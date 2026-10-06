def format_count(count: int, singular: str, plural: str | None = None) -> str:
    """
    Format a count with singular/plural wording.

    Examples:
        1 page
        2 pages
        1 chunk
        5 chunks
    """

    if plural is None:
        plural = f"{singular}s"

    word = singular if count == 1 else plural

    return f"{count:,} {word}"


def truncate_text(text: str, max_chars: int = 250) -> str:
    """
    Truncate text for compact UI previews.

    Whitespace is normalised so previews look cleaner in labels and captions.
    """

    if max_chars < 4:
        raise ValueError("max_chars must be at least 4.")

    cleaned_text = " ".join(text.split())

    if len(cleaned_text) <= max_chars:
        return cleaned_text

    return cleaned_text[: max_chars - 3].rstrip() + "..."


def format_source_label(index: int, file_name: str, page_number: int) -> str:
    """
    Format a source label consistently across the UI.
    """

    page_text = str(page_number) if page_number > 0 else "unknown"

    return f"[Source {index}] {file_name}, page {page_text}"


def status_icon(is_complete: bool) -> str:
    """
    Return a simple status icon for pipeline steps.
    """

    return "✅" if is_complete else "⬜"


def build_pipeline_status_items(
    page_count: int,
    chunk_count: int,
    embedded_count: int,
    stored_count: int,
) -> list[str]:
    """
    Build human-readable pipeline status messages.
    """

    return [
        (
            f"{status_icon(page_count > 0)} **PDF extraction** — "
            f"{format_count(page_count, 'page')} with text"
        ),
        (
            f"{status_icon(chunk_count > 0)} **Text chunking** — "
            f"{format_count(chunk_count, 'chunk')}"
        ),
        (
            f"{status_icon(embedded_count > 0)} **Embedding generation** — "
            f"{format_count(embedded_count, 'embedded chunk')}"
        ),
        (
            f"{status_icon(stored_count > 0)} **Vector storage** — "
            f"{format_count(stored_count, 'stored chunk')}"
        ),
    ]
