from src.config import AppConfig


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
