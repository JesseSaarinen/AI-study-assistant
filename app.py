"""Streamlit entry point for the AI Study Assistant."""

import streamlit as st

from src.config import ensure_storage_dir, get_config
from ui.ask_tab import display_ask_tab
from ui.inspect_tab import display_inspect_tab
from ui.process_tab import display_process_tab
from ui.sidebar import display_sidebar
from ui.state import initialise_session_state


def main() -> None:
    st.set_page_config(
        page_title="AI Study Assistant",
        page_icon="📚",
        layout="wide",
    )

    initialise_session_state()

    try:
        config = get_config()
        ensure_storage_dir(config)
    except Exception as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()

    display_sidebar(config)

    st.title("📚 AI Study Assistant for University Notes")

    st.write(
        """
        Upload lecture notes as PDFs, turn them into searchable embeddings, and ask
        questions grounded in your uploaded material.
        """
    )

    process_tab, inspect_tab, ask_tab = st.tabs(
        [
            "1. Process notes",
            "2. Inspect data",
            "3. Ask questions",
        ]
    )

    with process_tab:
        display_process_tab(config)

    with inspect_tab:
        stored_count = display_inspect_tab(config)

    with ask_tab:
        display_ask_tab(config, stored_count)


if __name__ == "__main__":
    main()
