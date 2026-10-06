import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    """
    Application configuration loaded from environment variables.
    """

    openai_api_key: str
    openai_base_url: str
    openai_model: str
    embedding_model_name: str
    chroma_db_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    chroma_collection_name: str = "study_notes"
    retrieval_distance_threshold: float | None = 1.0
    max_uploaded_file_mb: int = 50


def _get_int_env(name: str, default: int) -> int:
    """
    Read an integer environment variable.

    Raises:
        ValueError: If the environment variable exists but is not an integer.
    """

    raw_value = os.getenv(name)

    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer. Got: {raw_value}") from exc


def _get_optional_float_env(name: str, default: float | None) -> float | None:
    """
    Read an optional float environment variable.

    The value can be disabled by setting it to:
        disabled
        none
        null
        off
    """

    raw_value = os.getenv(name)

    if raw_value is None or raw_value.strip() == "":
        return default

    cleaned_value = raw_value.strip().lower()

    if cleaned_value in {"disabled", "none", "null", "off"}:
        return None

    try:
        return float(cleaned_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float or 'disabled'. Got: {raw_value}") from exc


def _validate_config_values(
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    retrieval_distance_threshold: float | None,
    max_uploaded_file_mb: int,
) -> None:
    """
    Validate numeric configuration values.
    """

    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("CHUNK_OVERLAP cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    if top_k <= 0:
        raise ValueError("TOP_K must be greater than 0.")

    if retrieval_distance_threshold is not None and retrieval_distance_threshold < 0:
        raise ValueError("RETRIEVAL_DISTANCE_THRESHOLD cannot be negative.")

    if max_uploaded_file_mb <= 0:
        raise ValueError("MAX_UPLOADED_FILE_MB must be greater than 0.")


def get_config() -> AppConfig:
    """
    Load application configuration from .env and environment variables.
    """

    load_dotenv()

    chunk_size = _get_int_env("CHUNK_SIZE", 1000)
    chunk_overlap = _get_int_env("CHUNK_OVERLAP", 150)
    top_k = _get_int_env("TOP_K", 5)
    retrieval_distance_threshold = _get_optional_float_env(
        "RETRIEVAL_DISTANCE_THRESHOLD",
        1.0,
    )
    max_uploaded_file_mb = _get_int_env("MAX_UPLOADED_FILE_MB", 50)

    _validate_config_values(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        retrieval_distance_threshold=retrieval_distance_threshold,
        max_uploaded_file_mb=max_uploaded_file_mb,
    )

    chroma_collection_name = os.getenv(
        "CHROMA_COLLECTION_NAME",
        "study_notes",
    ).strip()

    return AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2").strip(),
        chroma_db_dir=Path(os.getenv("CHROMA_DB_DIR", "storage/chroma")),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        chroma_collection_name=chroma_collection_name or "study_notes",
        retrieval_distance_threshold=retrieval_distance_threshold,
        max_uploaded_file_mb=max_uploaded_file_mb,
    )


def ensure_storage_dir(config: AppConfig) -> None:
    """
    Ensure the ChromaDB storage directory exists.
    """

    Path(config.chroma_db_dir).mkdir(parents=True, exist_ok=True)


def has_api_key(config: AppConfig) -> bool:
    """
    Return True if an API key has been configured.
    """

    return bool(config.openai_api_key)


def require_api_key(config: AppConfig) -> None:
    """
    Raise a clear error if the OpenAI-compatible API key is missing.
    """

    if not has_api_key(config):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Add it to your .env file before using answer generation."
        )
