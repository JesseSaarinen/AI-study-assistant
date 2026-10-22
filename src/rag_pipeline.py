from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.citations import build_source_citations, ensure_answer_has_citation_markers
from src.embeddings import EmbeddingError, EmbeddingModel, generate_embeddings
from src.models import ChatTurn, RagAnswer, RetrievedChunk
from src.prompts import ANSWER_NOT_FOUND, build_rag_messages, build_rewrite_messages
from src.vector_store import InvalidEmbeddingError, VectorStoreError, retrieve_similar_chunks

# A rewritten question longer than this is almost certainly an answer or an
# explanation rather than a question, so the original question is used instead.
MAX_REWRITTEN_QUESTION_CHARS = 500


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
    ) -> str: ...

    def stream_answer(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
    ) -> Iterator[str]: ...


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


def describe_api_error(exc: Exception) -> str:
    """
    Turn an OpenAI client exception into a short message a user can act on.

    Free API tiers often hit rate limits, and a wrong model name or base URL is
    the most common setup mistake, so those cases get specific advice.
    """

    if isinstance(exc, openai.AuthenticationError):
        return "The API key was rejected. Check OPENAI_API_KEY."

    if isinstance(exc, openai.PermissionDeniedError):
        return "The API key is not allowed to use this model."

    if isinstance(exc, openai.RateLimitError):
        return "The chat API rate limit or quota was reached. Wait a minute and try again."

    if isinstance(exc, openai.NotFoundError):
        return "The chat model was not found. Check OPENAI_MODEL and OPENAI_BASE_URL."

    if isinstance(exc, openai.APITimeoutError):
        return "The chat API did not respond in time. Try again."

    if isinstance(exc, openai.APIConnectionError):
        return "Could not connect to the chat API. Check OPENAI_BASE_URL and your connection."

    if isinstance(exc, openai.APIStatusError):
        return f"The chat API returned an error (HTTP {exc.status_code})."

    return "Chat model request failed."


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
                timeout=60.0,
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
                messages=cast(list[ChatCompletionMessageParam], messages),
                temperature=temperature,
                max_tokens=700,
            )
        except Exception as exc:
            raise ChatModelError(describe_api_error(exc)) from exc

        try:
            content = response.choices[0].message.content
        except Exception as exc:
            raise ChatModelError("Chat model returned an unexpected response format.") from exc

        return content or ""

    def stream_answer(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """
        Stream an answer from the chat model as text fragments.
        """

        if not model.strip():
            raise ChatModelError("Chat model name cannot be empty.")

        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=cast(list[ChatCompletionMessageParam], messages),
                temperature=temperature,
                max_tokens=700,
                stream=True,
            )
        except Exception as exc:
            raise ChatModelError(describe_api_error(exc)) from exc

        try:
            for chunk in stream:
                # Some providers send a final chunk with usage data and no choices.
                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content

                if content:
                    yield content
        except Exception as exc:
            raise ChatModelError(
                f"The answer stream was interrupted. {describe_api_error(exc)}"
            ) from exc


@dataclass(frozen=True)
class PreparedAnswer:
    """
    Everything the chat model needs to answer one question.

    messages is empty when no chunk was close enough to the question, in which
    case the chat model is not called at all.
    """

    question: str
    search_query: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)

    @property
    def needs_chat_model(self) -> bool:
        return bool(self.messages)


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


def clean_rewritten_question(rewritten: str) -> str:
    """
    Tidy the chat model's rewritten question.

    Models sometimes wrap the question in quotes or add a label such as
    "Standalone question:", even when told not to.
    """

    lines = [line.strip() for line in rewritten.strip().splitlines() if line.strip()]

    if not lines:
        return ""

    cleaned = lines[0]

    for label in ("standalone question:", "rewritten question:", "question:"):
        if cleaned.lower().startswith(label):
            cleaned = cleaned[len(label) :].strip()

    return cleaned.strip("\"'“”‘’` ").strip()


def rewrite_follow_up_question(
    question: str,
    history: list[ChatTurn],
    chat_model: ChatModel,
    chat_model_name: str,
) -> str:
    """
    Rewrite a follow-up question into a standalone question for retrieval.

    A question such as "why is that?" has nothing to search for on its own.
    The chat model rewrites it using the earlier turns. Without history, or if
    the rewrite fails or looks wrong, the original question is returned, so a
    rewrite problem never blocks an answer.
    """

    cleaned_question = question.strip()

    if not history:
        return cleaned_question

    try:
        rewritten = chat_model.generate_answer(
            messages=build_rewrite_messages(question=cleaned_question, history=history),
            model=chat_model_name,
            temperature=0.0,
        )
    except Exception:
        return cleaned_question

    cleaned_rewrite = clean_rewritten_question(rewritten)

    if not cleaned_rewrite or len(cleaned_rewrite) > MAX_REWRITTEN_QUESTION_CHARS:
        return cleaned_question

    return cleaned_rewrite


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


def prepare_answer(
    question: str,
    collection: Any,
    embedding_model: EmbeddingModel,
    chat_model: ChatModel,
    chat_model_name: str,
    top_k: int = 5,
    max_retrieval_distance: float | None = None,
    history: list[ChatTurn] | None = None,
) -> PreparedAnswer:
    """
    Run every step before answer generation.

    Steps:
    1. Rewrite a follow-up question into a standalone question, if there is history.
    2. Embed the standalone question and retrieve relevant chunks from ChromaDB.
    3. Filter weak matches by distance if configured.
    4. Build a grounded prompt using those chunks.

    Earlier turns are used only for the rewrite. The answer prompt contains the
    standalone question and freshly retrieved chunks, so every [Source N] label
    the model sees belongs to this answer.
    """

    cleaned_question = question.strip()

    if not cleaned_question:
        raise EmptyQuestionError("Question cannot be empty.")

    search_query = rewrite_follow_up_question(
        question=cleaned_question,
        history=history or [],
        chat_model=chat_model,
        chat_model_name=chat_model_name,
    )

    retrieved_chunks = retrieve_relevant_chunks_for_question(
        question=search_query,
        collection=collection,
        embedding_model=embedding_model,
        top_k=top_k,
        max_retrieval_distance=max_retrieval_distance,
    )

    if not retrieved_chunks:
        return PreparedAnswer(question=cleaned_question, search_query=search_query)

    messages = build_rag_messages(
        question=search_query,
        chunks=retrieved_chunks,
    )

    return PreparedAnswer(
        question=cleaned_question,
        search_query=search_query,
        sources=retrieved_chunks,
        messages=messages,
    )


def stream_prepared_answer(
    prepared: PreparedAnswer,
    chat_model: ChatModel,
    chat_model_name: str,
) -> Iterator[str]:
    """
    Stream the chat model's answer for a prepared question.

    Yields nothing when no relevant chunks were retrieved.
    """

    if not prepared.needs_chat_model:
        return

    try:
        yield from chat_model.stream_answer(
            messages=prepared.messages,
            model=chat_model_name,
            temperature=0.0,
        )
    except ChatModelError:
        raise
    except Exception as exc:
        raise ChatModelError("Chat model failed to generate an answer.") from exc


def finalise_answer(raw_answer: str, prepared: PreparedAnswer) -> RagAnswer:
    """
    Turn the raw model output into the answer shown to the user.

    Empty answers and the model's own not-found reply become the standard
    not-found message without sources. Otherwise structured citations are built
    and missing citation markers are added.
    """

    not_found = RagAnswer(
        answer=ANSWER_NOT_FOUND,
        sources=[],
        citations=[],
        search_query=prepared.search_query,
    )

    if not prepared.sources:
        return not_found

    answer = normalise_answer(raw_answer)

    if not answer or answer_is_not_found(answer):
        return not_found

    citations = build_source_citations(prepared.sources)
    answer_with_citations = ensure_answer_has_citation_markers(
        answer=answer,
        citations=citations,
    )

    return RagAnswer(
        answer=answer_with_citations,
        sources=prepared.sources,
        citations=citations,
        search_query=prepared.search_query,
    )


def answer_question(
    question: str,
    collection: Any,
    embedding_model: EmbeddingModel,
    chat_model: ChatModel,
    chat_model_name: str,
    top_k: int = 5,
    max_retrieval_distance: float | None = None,
    history: list[ChatTurn] | None = None,
) -> RagAnswer:
    """
    Run the full RAG question-answering pipeline without streaming.

    See prepare_answer for the retrieval steps. The chat model is then asked to
    answer using only the retrieved context, and the answer is returned with its
    source chunks and structured citations.
    """

    prepared = prepare_answer(
        question=question,
        collection=collection,
        embedding_model=embedding_model,
        chat_model=chat_model,
        chat_model_name=chat_model_name,
        top_k=top_k,
        max_retrieval_distance=max_retrieval_distance,
        history=history,
    )

    if not prepared.needs_chat_model:
        return finalise_answer("", prepared)

    try:
        raw_answer = chat_model.generate_answer(
            messages=prepared.messages,
            model=chat_model_name,
            temperature=0.0,
        )
    except ChatModelError:
        raise
    except Exception as exc:
        raise ChatModelError("Chat model failed to generate an answer.") from exc

    return finalise_answer(raw_answer, prepared)
