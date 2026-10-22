import streamlit as st

from src.citations import extract_source_numbers, format_citation, get_citations_to_display
from src.config import AppConfig, has_api_key
from src.models import RagAnswer
from src.rag_pipeline import ChatModelError, EmptyQuestionError, RagPipelineError, answer_question
from src.ui_helpers import format_source_label, truncate_text
from src.vector_store import get_or_create_collection
from ui.resources import get_cached_chat_model, get_cached_chroma_client, get_cached_embedding_model


def display_rag_answer(rag_answer: RagAnswer | None) -> None:
    """
    Display the generated RAG answer, structured citations, and source chunks.
    """

    if rag_answer is None:
        return

    st.subheader("Answer")
    st.markdown(rag_answer.answer)

    if not rag_answer.sources:
        return

    if rag_answer.citations:
        st.subheader("Citations")

        citations_to_display = get_citations_to_display(
            answer=rag_answer.answer,
            citations=rag_answer.citations,
        )

        for citation in citations_to_display:
            st.markdown(f"- `{format_citation(citation)}`")

    cited_source_numbers = set(extract_source_numbers(rag_answer.answer))

    st.subheader("Retrieved source chunks")

    for index, source in enumerate(rag_answer.sources, start=1):
        source_label = format_source_label(
            index=index,
            file_name=source.file_name,
            page_number=source.page_number,
        )

        if index in cited_source_numbers:
            source_label = f"{source_label} ✅ cited"

        distance_text = (
            f"Similarity distance: `{source.distance:.4f}`"
            if source.distance is not None
            else "Similarity distance: unavailable"
        )

        with st.expander(source_label, expanded=index == 1):
            st.caption(f"Chunk ID: `{source.chunk_id}`")
            st.caption(distance_text)
            st.caption(f"Preview: {truncate_text(source.text, max_chars=180)}")

            st.text_area(
                f"Source {index} text",
                value=source.text,
                height=180,
                disabled=True,
                key=f"rag_source_text_{index}_{source.chunk_id}",
            )


def display_ask_tab(config: AppConfig, stored_count: int) -> None:
    """
    Display the question-answering tab.
    """

    st.header("3. Ask questions")

    st.write(
        """
        Ask a question about the notes currently stored in ChromaDB. The assistant
        will retrieve relevant chunks and answer using only those chunks.
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

    with st.form("question_form"):
        question = st.text_input(
            "Question",
            placeholder="Example: What is dynamic programming?",
            value=st.session_state.last_question,
        )

        submitted = st.form_submit_button("Ask question", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Retrieving relevant notes and generating answer..."):
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

                    rag_answer = answer_question(
                        question=question,
                        collection=collection,
                        embedding_model=embedding_model,
                        chat_model=chat_model,
                        chat_model_name=config.openai_model,
                        top_k=config.top_k,
                        max_retrieval_distance=config.retrieval_distance_threshold,
                    )

                st.session_state.rag_answer = rag_answer
                st.session_state.last_question = question

            except EmptyQuestionError as exc:
                st.warning(str(exc))

            except ChatModelError as exc:
                st.error(f"Chat model error: {exc}")

            except RagPipelineError as exc:
                st.error(f"RAG pipeline error: {exc}")

            except Exception as exc:
                st.error(f"Unexpected question-answering error: {exc}")

    if st.button("Clear answer"):
        st.session_state.rag_answer = None
        st.session_state.last_question = ""

    display_rag_answer(st.session_state.rag_answer)
