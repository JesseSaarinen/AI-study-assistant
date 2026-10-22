import pytest

from src.config import DEFAULT_RETRIEVAL_DISTANCE_THRESHOLD, AppConfig, get_config


def test_app_config_can_be_created() -> None:
    config = AppConfig(
        openai_api_key="test-key",
        openai_base_url="https://api.openai.com/v1",
        openai_model="gpt-4o-mini",
        embedding_model_name="all-MiniLM-L6-v2",
        chroma_db_dir="storage/chroma",
        chunk_size=1000,
        chunk_overlap=150,
        top_k=5,
    )

    assert config.openai_api_key == "test-key"
    assert config.chunk_size == 1000
    assert config.chunk_overlap == 150
    assert config.top_k == 5


def test_chunk_overlap_is_smaller_than_chunk_size() -> None:
    config = AppConfig(
        openai_api_key="test-key",
        openai_base_url="https://api.openai.com/v1",
        openai_model="gpt-4o-mini",
        embedding_model_name="all-MiniLM-L6-v2",
        chroma_db_dir="storage/chroma",
        chunk_size=1000,
        chunk_overlap=150,
        top_k=5,
    )

    assert config.chunk_overlap < config.chunk_size


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


def test_get_config_demo_mode_is_off_by_default(monkeypatch) -> None:
    set_base_env(monkeypatch)
    monkeypatch.delenv("DEMO_MODE", raising=False)

    config = get_config()

    assert config.demo_mode is False
    assert config.demo_max_questions == 20


@pytest.mark.parametrize("raw_value", ["true", "TRUE", "yes", "1", "on"])
def test_get_config_reads_demo_mode(monkeypatch, raw_value: str) -> None:
    set_base_env(monkeypatch)
    monkeypatch.setenv("DEMO_MODE", raw_value)
    monkeypatch.setenv("DEMO_MAX_QUESTIONS", "5")

    config = get_config()

    assert config.demo_mode is True
    assert config.demo_max_questions == 5


def test_get_config_rejects_invalid_demo_mode(monkeypatch) -> None:
    set_base_env(monkeypatch)
    monkeypatch.setenv("DEMO_MODE", "maybe")

    with pytest.raises(ValueError):
        get_config()


def test_get_config_rejects_invalid_demo_max_questions(monkeypatch) -> None:
    set_base_env(monkeypatch)
    monkeypatch.setenv("DEMO_MAX_QUESTIONS", "0")

    with pytest.raises(ValueError):
        get_config()


def test_get_config_uses_evaluated_default_threshold(monkeypatch) -> None:
    set_base_env(monkeypatch)
    monkeypatch.delenv("RETRIEVAL_DISTANCE_THRESHOLD", raising=False)

    config = get_config()

    assert config.retrieval_distance_threshold == DEFAULT_RETRIEVAL_DISTANCE_THRESHOLD


def test_get_config_reads_chat_history_turns(monkeypatch) -> None:
    set_base_env(monkeypatch)
    monkeypatch.delenv("CHAT_HISTORY_TURNS", raising=False)

    assert get_config().chat_history_turns == 3

    monkeypatch.setenv("CHAT_HISTORY_TURNS", "0")

    assert get_config().chat_history_turns == 0


def test_get_config_rejects_negative_chat_history_turns(monkeypatch) -> None:
    set_base_env(monkeypatch)
    monkeypatch.setenv("CHAT_HISTORY_TURNS", "-1")

    with pytest.raises(ValueError):
        get_config()
