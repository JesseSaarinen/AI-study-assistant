import streamlit as st

from src.embeddings import load_embedding_model
from src.rag_pipeline import OpenAIChatModel
from src.vector_store import get_chroma_client


@st.cache_resource(show_spinner=False)
def get_cached_embedding_model(model_name: str):
    """
    Load and cache the embedding model.

    Streamlit reruns the app after interactions. Caching prevents repeatedly
    loading the Sentence Transformers model.
    """

    return load_embedding_model(model_name)


@st.cache_resource(show_spinner=False)
def get_cached_chroma_client(chroma_db_dir: str):
    """
    Load and cache the ChromaDB client.
    """

    return get_chroma_client(chroma_db_dir)


@st.cache_resource(show_spinner=False)
def get_cached_chat_model(api_key: str, base_url: str):
    """
    Load and cache the OpenAI-compatible chat client.
    """

    return OpenAIChatModel(api_key=api_key, base_url=base_url)
