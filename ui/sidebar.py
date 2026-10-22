import streamlit as st

from src.config import AppConfig, has_api_key
from ui.state import clear_working_state


def display_sidebar(config: AppConfig) -> None:
    """
    Display app status and settings in the sidebar.
    """

    with st.sidebar:
        st.header("📚 AI Study Assistant")

        st.caption("RAG-powered question answering for university notes.")

        st.divider()

        if config.demo_mode:
            st.info(
                "Public demo: notes you process are stored only for your browser session, "
                f"with up to {config.demo_max_questions} questions or quizzes per session."
            )

        if has_api_key(config):
            st.success("API key found")
        else:
            st.warning("API key missing")

        st.subheader("Settings")
        st.write(f"Embedding model: `{config.embedding_model_name}`")
        st.write(f"Chat model: `{config.openai_model}`")
        st.write(f"Chunk size: `{config.chunk_size}`")
        st.write(f"Chunk overlap: `{config.chunk_overlap}`")
        st.write(f"Top K retrieval: `{config.top_k}`")
        st.write(f"Follow-up history: `{config.chat_history_turns}` turn(s)")

        threshold_display = (
            "disabled"
            if config.retrieval_distance_threshold is None
            else f"`{config.retrieval_distance_threshold}`"
        )

        st.write(f"Retrieval distance threshold: {threshold_display}")
        st.write(f"Max uploaded PDF size: `{config.max_uploaded_file_mb} MB`")
        st.write(f"ChromaDB directory: `{config.chroma_db_dir}`")
        st.write(f"Collection: `{config.chroma_collection_name}`")

        st.divider()

        if st.button("Clear current session state"):
            clear_working_state()
            st.success("Session state cleared.")
