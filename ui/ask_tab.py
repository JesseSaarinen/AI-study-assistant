from collections.abc import Iterator

import streamlit as st

from src.citations import extract_source_numbers, format_citation, get_citations_to_display
from src.config import AppConfig, has_api_key
from src.models import ChatTurn, RagAnswer
from src.rag_pipeline import (
    ChatModelError,
    EmptyQuestionError,
    RagPipelineError,
    finalise_answer,
    prepare_answer,
    stream_prepared_answer,
)
from src.ui_helpers import format_source_label
from src.vector_store import get_or_create_collection
from ui.resources import get_cached_chat_model, get_cached_chroma_client, get_cached_embedding_model
from ui.sample_notes import SAMPLE_QUESTIONS
from ui.state import clear_conversation

STREAMING_CURSOR = "▌"


def display_search_query(question: str, rag_answer: RagAnswer) -> None:
    """
    Show the standalone question used for retrieval when a follow-up was rewritten.
    """

    search_query = rag_answer.search_query.strip()

    if search_query and search_query.lower() != question.strip().lower():
        st.caption(f"🔎 Searched the notes for: {search_query}")


def display_sources(rag_answer: RagAnswer, turn_index: int) -> None:
    """
    Show the citations and retrieved chunks behind one answer.
    """

    if not rag_answer.sources:
        return

    cited_source_numbers = set(extract_source_numbers(rag_answer.answer))
    cited_count = sum(
        1 for number in range(1, len(rag_answer.sources) + 1) if number in cited_source_numbers
    )

    with st.expander(f"Sources: {len(rag_answer.sources)} retrieved, {cited_count} cited"):
        if rag_answer.citations:
            citations_to_display = get_citations_to_display(
                answer=rag_answer.answer,
                citations=rag_answer.citations,
            )

            for citation in citations_to_display:
                st.markdown(f"- `{format_citation(citation)}`")

        tab_labels = [
            f"Source {index}{' ✅' if index in cited_source_numbers else ''}"
            for index in range(1, len(rag_answer.sources) + 1)
        ]

        for index, (tab, source) in enumerate(
            zip(st.tabs(tab_labels), rag_answer.sources, strict=True),
            start=1,
        ):
            with tab:
                st.markdown(
                    f"**{format_source_label(index, source.file_name, source.page_number)}**"
                )

                distance_text = (
                    f"Similarity distance `{source.distance:.4f}`"
                    if source.distance is not None
                    else "Similarity distance unavailable"
                )
                st.caption(f"{distance_text} · Chunk ID `{source.chunk_id}`")

                st.text_area(
                    f"Source {index} text",
                    value=source.text,
                    height=160,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"source_text_{turn_index}_{index}_{source.chunk_id}",
                )


def display_turn(turn: ChatTurn, turn_index: int) -> None:
    """
    Display one earlier question and its answer.
    """

    with st.chat_message("user"):
        st.markdown(turn.question)

    with st.chat_message("assistant"):
        display_search_query(turn.question, turn.answer)
        st.markdown(turn.answer.answer)
        display_sources(turn.answer, turn_index)


def stream_into_placeholder(placeholder, fragments: Iterator[str]) -> str:
    """
    Write streamed text fragments into a placeholder as they arrive.

    Returns:
        The complete streamed text.
    """

    text = ""

    try:
        for fragment in fragments:
            text += fragment
            placeholder.markdown(text + STREAMING_CURSOR)
    except Exception:
        placeholder.empty()
        raise

    return text


def generate_answer(question: str, history: list[ChatTurn], config: AppConfig) -> RagAnswer:
    """
    Retrieve context for a question and stream the answer into the current container.
    """

    with st.spinner("Searching your notes..."):
        client = get_cached_chroma_client(str(config.chroma_db_dir))
        collection = get_or_create_collection(
            client=client,
            collection_name=config.chroma_collection_name,
        )

        embedding_model = get_cached_embedding_model(config.embedding_model_name)

        chat_model = get_cached_chat_model(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
        )

        prepared = prepare_answer(
            question=question,
            collection=collection,
            embedding_model=embedding_model,
            chat_model=chat_model,
            chat_model_name=config.openai_model,
            top_k=config.top_k,
            max_retrieval_distance=config.retrieval_distance_threshold,
            history=history,
        )

    answer_placeholder = st.empty()

    raw_answer = stream_into_placeholder(
        answer_placeholder,
        stream_prepared_answer(
            prepared=prepared,
            chat_model=chat_model,
            chat_model_name=config.openai_model,
        ),
    )

    rag_answer = finalise_answer(raw_answer, prepared)

    with answer_placeholder.container():
        display_search_query(question, rag_answer)
        st.markdown(rag_answer.answer)

    return rag_answer


def ask_question(question: str, config: AppConfig) -> None:
    """
    Answer one question, show it as the newest chat turn, and add it to the history.
    """

    history: list[ChatTurn] = st.session_state.chat_history
    turn_index = len(history)
    recent_history = history[-config.chat_history_turns :] if config.chat_history_turns else []

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            rag_answer = generate_answer(question, recent_history, config)

        except EmptyQuestionError as exc:
            st.warning(str(exc))
            return

        except ChatModelError as exc:
            st.error(f"Chat model error: {exc}")
            return

        except RagPipelineError as exc:
            st.error(f"RAG pipeline error: {exc}")
            return

        except Exception as exc:
            st.error(f"Unexpected question-answering error: {exc}")
            return

        display_sources(rag_answer, turn_index)

    st.session_state.chat_history.append(ChatTurn(question=question, answer=rag_answer))
    st.session_state.questions_asked += 1


def display_example_questions() -> str | None:
    """
    Show one-click example questions when the sample notes are loaded.

    Returns:
        The clicked example question, if any.
    """

    if not st.session_state.sample_notes_loaded:
        return None

    st.caption("Try an example question about the sample notes:")

    clicked_question = None

    with st.container(horizontal=True):
        for sample_question in SAMPLE_QUESTIONS:
            if st.button(sample_question):
                clicked_question = sample_question

    return clicked_question


def demo_question_limit_reached(config: AppConfig) -> bool:
    """
    Return True when a public demo session has used up its question budget.
    """

    return config.demo_mode and st.session_state.questions_asked >= config.demo_max_questions


def display_ask_tab(config: AppConfig, stored_count: int) -> None:
    """
    Display the question-answering tab.
    """

    st.header("3. Ask questions")

    st.write(
        """
        Ask a question about the notes currently stored in ChromaDB. The assistant
        retrieves relevant chunks and answers using only those chunks. Follow-up
        questions such as "why is that?" work too.
        """
    )

    if stored_count == 0:
        st.info("Process and store notes in ChromaDB before asking questions.")
        return

    if not has_api_key(config):
        st.warning("Add `OPENAI_API_KEY` to your `.env` file before asking questions.")
        return

    if config.retrieval_distance_threshold is not None:
        st.caption(
            f"Retrieval filtering is enabled. Chunks with distance above "
            f"`{config.retrieval_distance_threshold}` will be ignored."
        )
    else:
        st.caption("Retrieval distance filtering is disabled.")

    example_question = display_example_questions()

    # Created before the chat input so that new turns appear above the input.
    conversation = st.container()

    typed_question = st.chat_input(
        "Ask about your notes, or ask a follow-up question",
        key="question_input",
    )

    question = example_question or typed_question

    with conversation:
        for turn_index, turn in enumerate(st.session_state.chat_history):
            display_turn(turn, turn_index)

        if question and question.strip():
            if demo_question_limit_reached(config):
                st.warning(
                    f"You have reached the demo limit of {config.demo_max_questions} "
                    "question(s) for this session. Run the app locally with your own API key "
                    "to keep going."
                )
            else:
                ask_question(question.strip(), config)

    if st.session_state.chat_history:
        st.button("Clear conversation", on_click=clear_conversation)
