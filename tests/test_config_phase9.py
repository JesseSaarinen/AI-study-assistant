import pytest

from src.config import get_config


def set_base_env(monkeypatch) -> None:
    """
    Set a known-good environment for config tests.
    """

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    monkeypatch.setenv("CHROMA_DB_DIR", "storage/chroma")
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "study_notes")
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP", "150")
    monkeypatch.setenv("TOP_K", "5")
    monkeypatch.setenv("MAX_UPLOADED_FILE_MB", "50")


def test_get_config_reads_retrieval_distance_threshold(monkeypatch) -> None:
    set_base_env(monkeypatch)
    monkeypatch.setenv("RETRIEVAL_DISTANCE_THRESHOLD", "0.85")

    config = get_config()

    assert config.retrieval_distance_threshold == 0.85


def test_get_config_allows_disabled_retrieval_distance_threshold(monkeypatch) -> None:
    set_base_env(monkeypatch)
    monkeypatch.setenv("RETRIEVAL_DISTANCE_THRESHOLD", "disabled")

    config = get_config()

    assert config.retrieval_distance_threshold is None


def test_get_config_rejects_negative_retrieval_distance_threshold(monkeypatch) -> None:
    set_base_env(monkeypatch)
    monkeypatch.setenv("RETRIEVAL_DISTANCE_THRESHOLD", "-0.1")

    with pytest.raises(ValueError):
        get_config()


def test_get_config_rejects_invalid_top_k(monkeypatch) -> None:
    set_base_env(monkeypatch)
    monkeypatch.setenv("TOP_K", "0")

    with pytest.raises(ValueError):
        get_config()


def test_get_config_rejects_invalid_max_uploaded_file_size(monkeypatch) -> None:
    set_base_env(monkeypatch)
    monkeypatch.setenv("MAX_UPLOADED_FILE_MB", "0")

    with pytest.raises(ValueError):
        get_config()
