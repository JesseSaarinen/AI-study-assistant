"""Streamlit entry point for the AI Study Assistant."""

import os
from dataclasses import replace

import streamlit as st

from src.config import ensure_storage_dir, get_config
from ui.ask_tab import display_ask_tab
from ui.inspect_tab import display_inspect_tab
from ui.process_tab import display_process_tab
from ui.sidebar import display_sidebar
from ui.state import initialise_session_state, session_collection_name


def load_streamlit_secrets() -> None:
    """
    Copy top-level Streamlit secrets into environment variables.

    On Streamlit Community Cloud, settings are entered as secrets rather than a
    .env file. Copying them lets get_config() read every setting the same way.
    Values that are already set in the environment win.
    """

    try:
        secrets = dict(st.secrets)
    except Exception:
        # No secrets.toml exists, which is the normal case for local runs.
        return

    for key, value in secrets.items():
        if isinstance(value, str | int | float | bool):
            os.environ.setdefault(key, str(value))


def main() -> None:
    st.set_page_config(
        page_title="AI Study Assistant",
        page_icon="📚",
        layout="wide",
    )

    initialise_session_state()

    load_streamlit_secrets()

    try:
        config = get_config()
        ensure_storage_dir(config)
    except Exception as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()

    if config.demo_mode:
        config = replace(
            config,
            chroma_collection_name=session_collection_name(config.chroma_collection_name),
        )

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
