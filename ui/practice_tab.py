import random
import uuid

import streamlit as st

from src.config import AppConfig, has_api_key
from src.quiz import (
    MAX_QUESTIONS,
    MIN_QUESTIONS,
    QuizGenerationError,
    QuizQuestion,
    generate_quiz,
    quiz_to_flashcards_csv,
)
from src.rag_pipeline import ChatModelError, RagPipelineError
from src.vector_store import get_or_create_collection
from ui.resources import get_cached_chat_model, get_cached_chroma_client, get_cached_embedding_model
from ui.state import demo_limit_reached

OPTION_LETTERS = "ABCDEF"
DEFAULT_QUESTION_COUNT = 5


def create_quiz(config: AppConfig, question_count: int, topic: str) -> None:
    """
    Generate a quiz from the stored notes and keep it in session state.
    """

    try:
        with st.spinner("Writing questions from your notes..."):
            client = get_cached_chroma_client(str(config.chroma_db_dir))
            collection = get_or_create_collection(
                client=client,
                collection_name=config.chroma_collection_name,
            )

            questions = generate_quiz(
                collection=collection,
                embedding_model=get_cached_embedding_model(config.embedding_model_name),
                chat_model=get_cached_chat_model(
                    api_key=config.openai_api_key,
                    base_url=config.openai_base_url,
                ),
                chat_model_name=config.openai_model,
                question_count=question_count,
                topic=topic,
                max_retrieval_distance=config.retrieval_distance_threshold,
                rng=random.Random(),
            )

    except QuizGenerationError as exc:
        st.warning(str(exc))
        return

    except ChatModelError as exc:
        st.error(f"Chat model error: {exc}")
        return

    except RagPipelineError as exc:
        st.error(f"Retrieval error: {exc}")
        return

    except Exception as exc:
        st.error(f"Unexpected quiz error: {exc}")
        return

    st.session_state.quiz = questions
    st.session_state.quiz_id = uuid.uuid4().hex[:8]
    st.session_state.quiz_checked = False
    st.session_state.questions_asked += 1


def answer_key(quiz_id: str, question_number: int) -> str:
    return f"quiz_{quiz_id}_{question_number}"


def mark_quiz_checked() -> None:
    st.session_state.quiz_checked = True


def display_feedback(question: QuizQuestion, selected_index: int | None) -> None:
    """
    Show whether one answer was right, with the explanation and the source page.
    """

    correct_label = f"{OPTION_LETTERS[question.correct_index]}. {question.correct_option}"

    if selected_index == question.correct_index:
        st.markdown(f"✅ Correct: **{correct_label}**")
    elif selected_index is None:
        st.markdown(f"⚪ Not answered. The answer is **{correct_label}**")
    else:
        st.markdown(f"❌ The answer is **{correct_label}**")

    source = f"{question.file_name}, page {question.page_number}"
    explanation = f"{question.explanation} " if question.explanation else ""
    st.caption(f"{explanation}Source: {source}")


def display_quiz(questions: list[QuizQuestion]) -> None:
    """
    Show the quiz as a form, with the score and feedback once it is checked.
    """

    quiz_id = st.session_state.quiz_id
    checked = st.session_state.quiz_checked

    if checked:
        score = sum(
            1
            for number, question in enumerate(questions, start=1)
            if st.session_state.get(answer_key(quiz_id, number)) == question.correct_index
        )
        st.subheader(f"Score: {score} / {len(questions)}")

    with st.form(f"quiz_form_{quiz_id}"):
        for number, question in enumerate(questions, start=1):
            st.markdown(f"**{number}. {question.question}**")

            selected_index = st.radio(
                f"Answer to question {number}",
                options=list(range(len(question.options))),
                format_func=lambda index, q=question: (
                    f"{OPTION_LETTERS[index]}. {q.options[index]}"
                ),
                index=None,
                key=answer_key(quiz_id, number),
                label_visibility="collapsed",
            )

            if checked:
                display_feedback(question, selected_index)

        st.form_submit_button("Check answers", type="primary", on_click=mark_quiz_checked)

    st.download_button(
        "Download as flashcards (CSV for Anki)",
        data=quiz_to_flashcards_csv(questions),
        file_name="flashcards.csv",
        mime="text/csv",
        on_click="ignore",
    )


def display_practice_tab(config: AppConfig, stored_count: int) -> None:
    """
    Display the practice tab, where quizzes are generated from the stored notes.
    """

    st.header("4. Practice")

    st.write(
        """
        Test yourself with multiple-choice questions written from your notes. Each
        answer comes with an explanation and the page it is based on, and the
        questions can be downloaded as flashcards.
        """
    )

    if stored_count == 0:
        st.info("Process and store notes in ChromaDB before generating a quiz.")
        return

    if not has_api_key(config):
        st.warning("Add `OPENAI_API_KEY` to your `.env` file before generating a quiz.")
        return

    with st.form("quiz_settings"):
        topic = st.text_input(
            "Topic (optional)",
            placeholder="Example: hash tables. Leave empty to cover all of the notes.",
        )
        question_count = st.slider(
            "Number of questions",
            min_value=MIN_QUESTIONS,
            max_value=MAX_QUESTIONS,
            value=DEFAULT_QUESTION_COUNT,
        )
        generate_clicked = st.form_submit_button("Generate quiz", type="primary")

    if generate_clicked:
        if demo_limit_reached(config):
            st.warning(
                f"You have reached the demo limit of {config.demo_max_questions} request(s) "
                "for this session. Run the app locally with your own API key to keep going."
            )
        else:
            create_quiz(config, question_count, topic.strip())

    if st.session_state.quiz:
        display_quiz(st.session_state.quiz)
