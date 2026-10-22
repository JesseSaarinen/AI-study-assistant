from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.helpers import FixedAnswerChatModel, KeywordFakeEmbeddingModel
from ui.sample_notes import SAMPLE_QUESTIONS

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """
    Run the Streamlit app against a temporary database with fake models.
    """

    monkeypatch.setenv("CHROMA_DB_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setattr(
        "ui.resources.load_embedding_model",
        lambda model_name: KeywordFakeEmbeddingModel(),
    )
    monkeypatch.setattr(
        "ui.resources.OpenAIChatModel",
        lambda api_key, base_url: FixedAnswerChatModel("Amortised analysis. [Source 1]"),
    )

    st.cache_resource.clear()
    yield monkeypatch
    st.cache_resource.clear()


def click_button(app: AppTest, label: str) -> AppTest:
    button = next(button for button in app.button if button.label == label)
    return button.click().run()


def test_app_renders_without_errors(app_env) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value.startswith("📚 AI Study Assistant")
    assert [tab.label for tab in app.tabs] == [
        "1. Process notes",
        "2. Inspect data",
        "3. Ask questions",
    ]


def test_sample_notes_can_be_loaded_and_queried(app_env) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    app = click_button(app, "Try with sample notes")

    assert not app.exception
    assert app.session_state.sample_notes_loaded
    assert len(app.session_state.extracted_pages) == 8

    app = click_button(app, SAMPLE_QUESTIONS[0])

    assert not app.exception
    assert app.session_state.last_question == SAMPLE_QUESTIONS[0]
    assert app.session_state.rag_answer.answer == "Amortised analysis. [Source 1]"
    assert app.session_state.questions_asked == 1


def test_demo_mode_limits_questions_per_session(app_env) -> None:
    app_env.setenv("DEMO_MODE", "true")
    app_env.setenv("DEMO_MAX_QUESTIONS", "1")

    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app = click_button(app, "Try with sample notes")
    app = click_button(app, SAMPLE_QUESTIONS[0])
    app = click_button(app, SAMPLE_QUESTIONS[1])

    assert not app.exception
    assert app.session_state.questions_asked == 1
    assert app.session_state.last_question == SAMPLE_QUESTIONS[0]
    assert any("demo limit" in warning.value for warning in app.warning)
