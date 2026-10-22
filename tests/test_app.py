import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.rag_pipeline import ChatModelError
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
        "4. Practice",
    ]


def test_sample_notes_can_be_loaded_and_queried(app_env) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    app = click_button(app, "Try with sample notes")

    assert not app.exception
    assert app.session_state.sample_notes_loaded
    assert len(app.session_state.extracted_pages) == 8

    app = click_button(app, SAMPLE_QUESTIONS[0])

    assert not app.exception
    history = app.session_state.chat_history
    assert len(history) == 1
    assert history[0].question == SAMPLE_QUESTIONS[0]
    assert history[0].answer.answer == "Amortised analysis. [Source 1]"
    assert history[0].answer.sources
    assert app.session_state.questions_asked == 1
    assert [message.name for message in app.chat_message] == ["user", "assistant"]


def test_follow_up_question_is_rewritten_before_retrieval(app_env) -> None:
    chat_model = FixedAnswerChatModel(
        "Because the array doubles. [Source 1]",
        rewrite="Why does a dynamic array double its capacity?",
    )
    app_env.setattr("ui.resources.OpenAIChatModel", lambda api_key, base_url: chat_model)

    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app = click_button(app, "Try with sample notes")
    app = click_button(app, SAMPLE_QUESTIONS[0])
    app = app.chat_input[0].set_value("Why is that?").run()

    assert not app.exception
    history = app.session_state.chat_history
    assert [turn.question for turn in history] == [SAMPLE_QUESTIONS[0], "Why is that?"]
    assert history[1].answer.search_query == "Why does a dynamic array double its capacity?"
    assert any("Searched the notes for" in caption.value for caption in app.caption)

    # First question: one answer call. Follow-up: one rewrite call, then one answer call.
    assert len(chat_model.calls) == 3
    rewrite_prompt = chat_model.calls[1][1]["content"]
    assert SAMPLE_QUESTIONS[0] in rewrite_prompt
    assert "Why is that?" in rewrite_prompt
    assert "Why does a dynamic array double its capacity?" in chat_model.calls[2][1]["content"]


def test_conversation_can_be_cleared(app_env) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app = click_button(app, "Try with sample notes")
    app = click_button(app, SAMPLE_QUESTIONS[0])
    app = click_button(app, "Clear conversation")

    assert not app.exception
    assert app.session_state.chat_history == []
    assert not app.chat_message


def test_chat_model_errors_are_shown_and_not_added_to_history(app_env) -> None:
    class FailingChatModel(FixedAnswerChatModel):
        def stream_answer(self, messages, model, temperature=0.0):
            raise ChatModelError("The chat API rate limit or quota was reached.")
            yield ""

    app_env.setattr(
        "ui.resources.OpenAIChatModel",
        lambda api_key, base_url: FailingChatModel("unused"),
    )

    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app = click_button(app, "Try with sample notes")
    app = click_button(app, SAMPLE_QUESTIONS[0])

    assert not app.exception
    assert any("rate limit" in error.value for error in app.error)
    assert app.session_state.chat_history == []
    assert app.session_state.questions_asked == 0


def test_demo_mode_limits_questions_per_session(app_env) -> None:
    app_env.setenv("DEMO_MODE", "true")
    app_env.setenv("DEMO_MAX_QUESTIONS", "1")

    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app = click_button(app, "Try with sample notes")
    app = click_button(app, SAMPLE_QUESTIONS[0])
    app = click_button(app, SAMPLE_QUESTIONS[1])

    assert not app.exception
    assert app.session_state.questions_asked == 1
    assert [turn.question for turn in app.session_state.chat_history] == [SAMPLE_QUESTIONS[0]]
    assert any("demo limit" in warning.value for warning in app.warning)


def test_practice_quiz_can_be_generated_and_checked(app_env) -> None:
    quiz = [
        {
            "question": "Why is appending to a dynamic array cheap on average?",
            "options": [
                "Resizes are rare",
                "Arrays are sorted",
                "Memory is free",
                "It uses a heap",
            ],
            "correct_option": "Resizes are rare",
            "explanation": "Doubling the capacity makes resizes rare.",
            "source": 1,
        },
        {
            "question": "What does LIFO describe?",
            "options": ["A stack", "A queue", "A hash table", "A heap"],
            "correct_option": "A stack",
            "explanation": "A stack removes the newest item first.",
            "source": 1,
        },
    ]
    chat_model = FixedAnswerChatModel("unused", quiz=json.dumps(quiz))
    app_env.setattr("ui.resources.OpenAIChatModel", lambda api_key, base_url: chat_model)

    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app = click_button(app, "Try with sample notes")
    app.slider[0].set_value(2)
    app = click_button(app, "Generate quiz")

    assert not app.exception
    questions = app.session_state.quiz
    assert [question.question for question in questions] == [item["question"] for item in quiz]
    assert app.session_state.questions_asked == 1

    quiz_id = app.session_state.quiz_id
    first, second = questions
    app.radio(key=f"quiz_{quiz_id}_1").set_value(first.correct_index)
    app.radio(key=f"quiz_{quiz_id}_2").set_value((second.correct_index + 1) % 4)
    app = click_button(app, "Check answers")

    assert not app.exception
    assert any(header.value == "Score: 1 / 2" for header in app.subheader)
    assert any("Source: dsa_lecture_notes.pdf, page" in caption.value for caption in app.caption)


def test_new_notes_clear_the_quiz(app_env) -> None:
    quiz = [
        {
            "question": "What does LIFO describe?",
            "options": ["A stack", "A queue", "A hash table", "A heap"],
            "correct_option": "A stack",
            "source": 1,
        }
    ]
    chat_model = FixedAnswerChatModel("unused", quiz=json.dumps(quiz))
    app_env.setattr("ui.resources.OpenAIChatModel", lambda api_key, base_url: chat_model)

    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app = click_button(app, "Try with sample notes")
    app = click_button(app, "Generate quiz")
    assert len(app.session_state.quiz) == 1

    app = click_button(app, "Try with sample notes")

    assert app.session_state.quiz == []
