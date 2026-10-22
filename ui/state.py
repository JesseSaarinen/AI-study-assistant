import uuid

import streamlit as st


def initialise_session_state() -> None:
    """
    Initialise Streamlit session state values used by the app.
    """

    defaults = {
        "extracted_pages": [],
        "pdf_extraction_errors": [],
        "text_chunks": [],
        "embedded_chunks": [],
        "vector_store_message": "",
        "chat_history": [],
        "questions_asked": 0,
        "sample_notes_loaded": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_working_state() -> None:
    """
    Clear in-memory Streamlit state.

    This does not delete the ChromaDB collection from disk.
    """

    st.session_state.extracted_pages = []
    st.session_state.pdf_extraction_errors = []
    st.session_state.text_chunks = []
    st.session_state.embedded_chunks = []
    st.session_state.vector_store_message = ""
    clear_conversation()


def clear_conversation() -> None:
    """
    Forget the questions and answers asked so far.

    Called when the stored notes change, because earlier answers may cite
    chunks that no longer exist.
    """

    st.session_state.chat_history = []


def session_collection_name(base_name: str) -> str:
    """
    Return a collection name that is unique to the current browser session.

    Used in demo mode so visitors of a shared deployment never see or delete
    each other's notes.
    """

    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:12]

    return f"{base_name}_{st.session_state.session_id}"
