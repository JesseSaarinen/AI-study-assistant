from typing import Any, Protocol

from openai import OpenAI

from src.citations import build_source_citations, ensure_answer_has_citation_markers
from src.embeddings import EmbeddingError, EmbeddingModel, generate_embeddings
from src.models import RagAnswer, RetrievedChunk
from src.prompts import ANSWER_NOT_FOUND, build_rag_messages
from src.vector_store import (
    InvalidEmbeddingError,
    VectorStoreError,
    retrieve_similar_chunks,
)


class ChatModel(Protocol):
    """
    Minimal chat model interface used by the RAG pipeline.

    This lets us test the pipeline with a fake chat model without calling a real
    API during pytest.
    """

    def generate_answer(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
    ) -> str:
        ...


class RagPipelineError(Exception):
    """
    Base exception for RAG pipeline errors.
    """


class EmptyQuestionError(RagPipelineError):
    """
    Raised when the user question is empty.
    """


class ChatModelError(RagPipelineError):
    """
    Raised when the chat model call fails.
    """


class OpenAIChatModel:
    """
    OpenAI-compatible chat model wrapper.

    This works with OpenAI and with many providers that expose an OpenAI-compatible
    API by changing OPENAI_BASE_URL.
    """

    def __init__(self, api_key: str, base_url: str) -> None:
        cleaned_api_key = api_key.strip()
        cleaned_base_url = base_url.strip() or "https://api.openai.com/v1"

        if not cleaned_api_key:
            raise ChatModelError(
                "Missing OPENAI_API_KEY. Add it to your .env file before asking questions."
            )

        try:
            self.client = OpenAI(
                api_key=cleaned_api_key,
                base_url=cleaned_base_url,
            )
        except Exception as exc:
            raise ChatModelError("Failed to initialise OpenAI-compatible client.") from exc

    def generate_answer(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate an answer from the chat model.
        """

        if not model.strip():
            raise ChatModelError("Chat model name cannot be empty.")

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=700,
            )
        except Exception as exc:
            raise ChatModelError("Chat model request failed.") from exc

        try:
            content = response.choices[0].message.content
        except Exception as exc:
            raise ChatModelError("Chat model returned an unexpected response format.") from exc

        return content or ""


def normalise_answer(answer: str) -> str:
    """
    Clean up the model answer.
    """

    return answer.strip()


def answer_is_not_found(answer: str) -> bool:
    """
    Return True if the model gave the required fallback answer.
    """

    cleaned = answer.strip().lower()
    required = ANSWER_NOT_FOUND.lower()

    return required in cleaned


def retrieve_relevant_chunks_for_question(
    question: str,
    collection: Any,
    embedding_model: EmbeddingModel,
    top_k: int,
    max_retrieval_distance: float | None = None,
) -> list[RetrievedChunk]:
    """
    Embed a question and retrieve semantically similar chunks from ChromaDB.

    If max_retrieval_distance is set, chunks with a larger distance are filtered
    out before the LLM sees them. This reduces hallucinations caused by weakly
    related context.
    """

    cleaned_question = question.strip()

    if not cleaned_question:
        raise EmptyQuestionError("Question cannot be empty.")

    if top_k <= 0:
        raise RagPipelineError("top_k must be greater than 0.")

    if max_retrieval_distance is not None and max_retrieval_distance < 0:
        raise RagPipelineError("max_retrieval_distance cannot be negative.")

    try:
        question_embedding = generate_embeddings(
            texts=[cleaned_question],
            model=embedding_model,
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
    except EmbeddingError as exc:
        raise RagPipelineError("Failed to generate embedding for the question.") from exc

    try:
        return retrieve_similar_chunks(
            collection=collection,
            query_embedding=question_embedding,
            top_k=top_k,
            max_distance=max_retrieval_distance,
        )
    except (VectorStoreError, InvalidEmbeddingError) as exc:
        raise RagPipelineError("Failed to retrieve relevant chunks from ChromaDB.") from exc


def answer_question(
    question: str,
    collection: Any,
    embedding_model: EmbeddingModel,
    chat_model: ChatModel,
    chat_model_name: str,
    top_k: int = 5,
    max_retrieval_distance: float | None = None,
) -> RagAnswer:
    """
    Run the full RAG question-answering pipeline.

    Steps:
    1. Embed the user's question.
    2. Retrieve relevant chunks from ChromaDB.
    3. Filter weak matches by distance if configured.
    4. Build a grounded prompt using those chunks.
    5. Ask the chat model to answer using only the retrieved context.
    6. Build structured citations for the retrieved sources.
    7. Return the answer, source chunks, and citations.
    """

    cleaned_question = question.strip()

    if not cleaned_question:
        raise EmptyQuestionError("Question cannot be empty.")

    retrieved_chunks = retrieve_relevant_chunks_for_question(
        question=cleaned_question,
        collection=collection,
        embedding_model=embedding_model,
        top_k=top_k,
        max_retrieval_distance=max_retrieval_distance,
    )

    if not retrieved_chunks:
        return RagAnswer(
            answer=ANSWER_NOT_FOUND,
            sources=[],
            citations=[],
        )

    messages = build_rag_messages(
        question=cleaned_question,
        chunks=retrieved_chunks,
    )

    try:
        raw_answer = chat_model.generate_answer(
            messages=messages,
            model=chat_model_name,
            temperature=0.0,
        )
    except ChatModelError:
        raise
    except Exception as exc:
        raise ChatModelError("Chat model failed to generate an answer.") from exc

    answer = normalise_answer(raw_answer)

    if not answer:
        return RagAnswer(
            answer=ANSWER_NOT_FOUND,
            sources=[],
            citations=[],
        )

    if answer_is_not_found(answer):
        return RagAnswer(
            answer=ANSWER_NOT_FOUND,
            sources=[],
            citations=[],
        )

    citations = build_source_citations(retrieved_chunks)
    answer_with_citations = ensure_answer_has_citation_markers(
        answer=answer,
        citations=citations,
    )

    return RagAnswer(
        answer=answer_with_citations,
        sources=retrieved_chunks,
        citations=citations,
    )