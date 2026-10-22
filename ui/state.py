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
        "rag_answer": None,
        "last_question": "",
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
    st.session_state.rag_answer = None
    st.session_state.last_question = ""
