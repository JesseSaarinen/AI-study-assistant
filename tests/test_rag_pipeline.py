from types import SimpleNamespace

import httpx
import openai
import pytest

from src.models import ChatTurn, RagAnswer
from src.prompts import ANSWER_NOT_FOUND
from src.rag_pipeline import (
    ChatModelError,
    EmptyQuestionError,
    OpenAIChatModel,
    answer_is_not_found,
    answer_question,
    clean_rewritten_question,
    describe_api_error,
    finalise_answer,
    normalise_answer,
    prepare_answer,
    rewrite_follow_up_question,
    stream_prepared_answer,
)
from src.vector_store import get_chroma_client, get_or_create_collection, upsert_embedded_chunks
from tests.helpers import FixedAnswerChatModel, make_embedded_chunk


class FakeEmbeddingModel:
    """
    Fake embedding model for tests.

    It maps recursion-related text close to [1, 0] and database-related text
    close to [0, 1].
    """

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> list[list[float]]:
        embeddings: list[list[float]] = []

        for sentence in sentences:
            lowered = sentence.lower()

            if "database" in lowered or "sql" in lowered:
                embeddings.append([0.0, 1.0])
            else:
                embeddings.append([1.0, 0.0])

        return embeddings


class FakeChatModel:
    """
    Fake chat model for tests.
    """

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.was_called = False
        self.last_messages: list[dict[str, str]] = []

    def generate_answer(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
    ) -> str:
        self.was_called = True
        self.last_messages = messages
        return self.answer


def test_normalise_answer_strips_whitespace() -> None:
    assert normalise_answer("  Test answer. \n") == "Test answer."


def test_answer_is_not_found_detects_required_fallback() -> None:
    assert answer_is_not_found(ANSWER_NOT_FOUND)
    assert answer_is_not_found(f"{ANSWER_NOT_FOUND}\n")


def test_answer_question_raises_for_empty_question(tmp_path) -> None:
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "rag_test_notes")

    with pytest.raises(EmptyQuestionError):
        answer_question(
            question="   ",
            collection=collection,
            embedding_model=FakeEmbeddingModel(),
            chat_model=FakeChatModel("Answer"),
            chat_model_name="fake-model",
            top_k=3,
        )


def test_answer_question_returns_fallback_when_collection_empty(tmp_path) -> None:
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "rag_test_notes")

    chat_model = FakeChatModel("This should not be called.")

    answer = answer_question(
        question="What is recursion?",
        collection=collection,
        embedding_model=FakeEmbeddingModel(),
        chat_model=chat_model,
        chat_model_name="fake-model",
        top_k=3,
    )

    assert answer.answer == ANSWER_NOT_FOUND
    assert answer.sources == []
    assert chat_model.was_called is False


def test_answer_question_retrieves_context_and_calls_chat_model(tmp_path) -> None:
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "rag_test_notes")

    chunks = [
        make_embedded_chunk(
            chunk_id="recursion_chunk",
            text="Recursion is when a function calls itself.",
            embedding=[1.0, 0.0],
            file_name="algorithms.pdf",
            page_number=4,
        ),
        make_embedded_chunk(
            chunk_id="database_chunk",
            text="A database stores structured information.",
            embedding=[0.0, 1.0],
            file_name="databases.pdf",
            page_number=2,
        ),
    ]

    upsert_embedded_chunks(collection, chunks)

    chat_model = FakeChatModel("Recursion is when a function calls itself. [Source 1]")

    answer = answer_question(
        question="What is recursion?",
        collection=collection,
        embedding_model=FakeEmbeddingModel(),
        chat_model=chat_model,
        chat_model_name="fake-model",
        top_k=1,
    )

    assert chat_model.was_called is True
    assert "function calls itself" in answer.answer
    assert len(answer.sources) == 1
    assert answer.sources[0].chunk_id == "recursion_chunk"
    assert answer.sources[0].file_name == "algorithms.pdf"
    assert answer.sources[0].page_number == 4

    user_message = chat_model.last_messages[1]["content"]

    assert "What is recursion?" in user_message
    assert "Recursion is when a function calls itself." in user_message
    assert "algorithms.pdf" in user_message
    assert "Page: 4" in user_message


def test_answer_question_returns_fallback_when_chat_model_says_not_found(
    tmp_path,
) -> None:
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "rag_test_notes")

    chunk = make_embedded_chunk(
        chunk_id="recursion_chunk",
        text="Recursion is when a function calls itself.",
        embedding=[1.0, 0.0],
    )

    upsert_embedded_chunks(collection, [chunk])

    chat_model = FakeChatModel(ANSWER_NOT_FOUND)

    answer = answer_question(
        question="What is quantum mechanics?",
        collection=collection,
        embedding_model=FakeEmbeddingModel(),
        chat_model=chat_model,
        chat_model_name="fake-model",
        top_k=1,
    )

    assert answer.answer == ANSWER_NOT_FOUND
    assert answer.sources == []


def test_answer_question_returns_fallback_when_retrieval_distance_too_high(
    tmp_path,
) -> None:
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "rag_robust_notes")

    chunk = make_embedded_chunk(
        chunk_id="recursion_chunk",
        text="Recursion is when a function calls itself.",
        embedding=[1.0, 0.0],
    )

    upsert_embedded_chunks(collection, [chunk])

    chat_model = FakeChatModel("This should not be used.")

    answer = answer_question(
        question="What is SQL?",
        collection=collection,
        embedding_model=FakeEmbeddingModel(),
        chat_model=chat_model,
        chat_model_name="fake-model",
        top_k=1,
        max_retrieval_distance=0.5,
    )

    assert answer.answer == ANSWER_NOT_FOUND
    assert answer.sources == []
    assert answer.citations == []
    assert chat_model.was_called is False


def store_recursion_and_database_chunks(tmp_path):
    client = get_chroma_client(tmp_path)
    collection = get_or_create_collection(client, "rag_chat_notes")
    upsert_embedded_chunks(
        collection,
        [
            make_embedded_chunk(
                chunk_id="recursion_chunk",
                text="Recursion is when a function calls itself.",
                embedding=[1.0, 0.0],
            ),
            make_embedded_chunk(
                chunk_id="database_chunk",
                text="SQL queries a relational database.",
                embedding=[0.0, 1.0],
            ),
        ],
    )
    return collection


def make_turn(question: str, answer: str) -> ChatTurn:
    return ChatTurn(question=question, answer=RagAnswer(answer=answer, sources=[]))


def test_rewrite_follow_up_question_skips_model_without_history() -> None:
    chat_model = FixedAnswerChatModel("unused")

    rewritten = rewrite_follow_up_question(
        question="  What is recursion?  ",
        history=[],
        chat_model=chat_model,
        chat_model_name="fake-model",
    )

    assert rewritten == "What is recursion?"
    assert chat_model.was_called is False


def test_rewrite_follow_up_question_uses_model_output() -> None:
    chat_model = FixedAnswerChatModel("unused", rewrite='"How does SQL query a database?"')

    rewritten = rewrite_follow_up_question(
        question="How does it query?",
        history=[make_turn("What is SQL?", "SQL queries a relational database.")],
        chat_model=chat_model,
        chat_model_name="fake-model",
    )

    assert rewritten == "How does SQL query a database?"


@pytest.mark.parametrize("bad_rewrite", ["", "   \n", "x" * 600])
def test_rewrite_follow_up_question_falls_back_to_original(bad_rewrite: str) -> None:
    chat_model = FixedAnswerChatModel("unused", rewrite=bad_rewrite)

    rewritten = rewrite_follow_up_question(
        question="Why?",
        history=[make_turn("What is SQL?", "A query language.")],
        chat_model=chat_model,
        chat_model_name="fake-model",
    )

    assert rewritten == "Why?"


def test_rewrite_follow_up_question_falls_back_when_model_fails() -> None:
    class BrokenChatModel(FixedAnswerChatModel):
        def generate_answer(self, messages, model, temperature=0.0):
            raise ChatModelError("Rate limit.")

    rewritten = rewrite_follow_up_question(
        question="Why?",
        history=[make_turn("What is SQL?", "A query language.")],
        chat_model=BrokenChatModel("unused"),
        chat_model_name="fake-model",
    )

    assert rewritten == "Why?"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("What is a heap?", "What is a heap?"),
        ("Standalone question: What is a heap?", "What is a heap?"),
        ("“What is a heap?”\nThis rewrite adds the topic.", "What is a heap?"),
        ("`What is a heap?`", "What is a heap?"),
    ],
)
def test_clean_rewritten_question(raw: str, expected: str) -> None:
    assert clean_rewritten_question(raw) == expected


def test_answer_question_retrieves_with_rewritten_follow_up(tmp_path) -> None:
    collection = store_recursion_and_database_chunks(tmp_path)
    chat_model = FixedAnswerChatModel(
        "SQL queries a relational database. [Source 1]",
        rewrite="How does SQL work?",
    )

    answer = answer_question(
        question="And how does that work?",
        collection=collection,
        embedding_model=FakeEmbeddingModel(),
        chat_model=chat_model,
        chat_model_name="fake-model",
        top_k=1,
        history=[make_turn("What is SQL?", "SQL is a query language. [Source 1]")],
    )

    assert answer.search_query == "How does SQL work?"
    assert answer.sources[0].chunk_id == "database_chunk"
    answer_prompt = chat_model.last_messages[1]["content"]
    assert "How does SQL work?" in answer_prompt
    assert "SQL is a query language" not in answer_prompt


def test_prepare_answer_without_matches_does_not_need_chat_model(tmp_path) -> None:
    collection = store_recursion_and_database_chunks(tmp_path)
    chat_model = FixedAnswerChatModel("unused")

    prepared = prepare_answer(
        question="What is SQL?",
        collection=collection,
        embedding_model=FakeEmbeddingModel(),
        chat_model=chat_model,
        chat_model_name="fake-model",
        top_k=1,
        max_retrieval_distance=0.5,
    )

    assert prepared.needs_chat_model is True

    empty_collection = get_or_create_collection(get_chroma_client(tmp_path), "empty_notes")

    prepared = prepare_answer(
        question="What is SQL?",
        collection=empty_collection,
        embedding_model=FakeEmbeddingModel(),
        chat_model=chat_model,
        chat_model_name="fake-model",
        top_k=1,
    )

    assert prepared.needs_chat_model is False
    assert prepared.messages == []
    assert list(stream_prepared_answer(prepared, chat_model, "fake-model")) == []
    assert finalise_answer("", prepared).answer == ANSWER_NOT_FOUND
    assert chat_model.was_called is False


def test_streamed_answer_is_finalised_with_citations(tmp_path) -> None:
    collection = store_recursion_and_database_chunks(tmp_path)
    chat_model = FixedAnswerChatModel("Recursion is when a function calls itself.")

    prepared = prepare_answer(
        question="What is recursion?",
        collection=collection,
        embedding_model=FakeEmbeddingModel(),
        chat_model=chat_model,
        chat_model_name="fake-model",
        top_k=1,
    )
    fragments = list(stream_prepared_answer(prepared, chat_model, "fake-model"))
    answer = finalise_answer("".join(fragments), prepared)

    assert len(fragments) > 1
    assert answer.answer.startswith("Recursion is when a function calls itself.")
    assert answer.answer.endswith("Sources: [Source 1]")
    assert answer.citations[0].chunk_id == "recursion_chunk"
    assert answer.search_query == "What is recursion?"


def test_finalise_answer_turns_model_not_found_into_fallback(tmp_path) -> None:
    collection = store_recursion_and_database_chunks(tmp_path)
    prepared = prepare_answer(
        question="What is recursion?",
        collection=collection,
        embedding_model=FakeEmbeddingModel(),
        chat_model=FixedAnswerChatModel("unused"),
        chat_model_name="fake-model",
        top_k=1,
    )

    for raw_answer in ["", "   ", f"Sorry. {ANSWER_NOT_FOUND}"]:
        answer = finalise_answer(raw_answer, prepared)

        assert answer.answer == ANSWER_NOT_FOUND
        assert answer.sources == []


def test_stream_prepared_answer_wraps_unexpected_errors(tmp_path) -> None:
    class ExplodingChatModel(FixedAnswerChatModel):
        def stream_answer(self, messages, model, temperature=0.0):
            yield "Partial"
            raise RuntimeError("socket closed")

    collection = store_recursion_and_database_chunks(tmp_path)
    prepared = prepare_answer(
        question="What is recursion?",
        collection=collection,
        embedding_model=FakeEmbeddingModel(),
        chat_model=FixedAnswerChatModel("unused"),
        chat_model_name="fake-model",
        top_k=1,
    )

    with pytest.raises(ChatModelError):
        list(stream_prepared_answer(prepared, ExplodingChatModel("unused"), "fake-model"))


def fake_stream_chunk(content: str | None, has_choices: bool = True):
    choices = [SimpleNamespace(delta=SimpleNamespace(content=content))] if has_choices else []
    return SimpleNamespace(choices=choices)


def make_openai_chat_model(create) -> OpenAIChatModel:
    chat_model = OpenAIChatModel(api_key="test-key", base_url="https://example.com/v1")
    chat_model.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return chat_model


def test_openai_chat_model_streams_text_fragments() -> None:
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return iter(
            [
                fake_stream_chunk("Recursion "),
                fake_stream_chunk(None),
                fake_stream_chunk("calls itself."),
                fake_stream_chunk(None, has_choices=False),
            ]
        )

    chat_model = make_openai_chat_model(create)
    messages = [{"role": "user", "content": "What is recursion?"}]

    fragments = list(chat_model.stream_answer(messages, model="fake-model"))

    assert fragments == ["Recursion ", "calls itself."]
    assert requests[0]["stream"] is True
    assert requests[0]["messages"] == messages


def make_status_error(error_class, status_code: int):
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return error_class("error", response=response, body=None)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (make_status_error(openai.AuthenticationError, 401), "API key was rejected"),
        (make_status_error(openai.RateLimitError, 429), "rate limit"),
        (make_status_error(openai.NotFoundError, 404), "OPENAI_MODEL"),
        (make_status_error(openai.InternalServerError, 503), "HTTP 503"),
        (
            openai.APIConnectionError(request=httpx.Request("POST", "https://example.com")),
            "Could not connect",
        ),
        (RuntimeError("boom"), "Chat model request failed."),
    ],
)
def test_openai_chat_model_explains_api_errors(error: Exception, expected: str) -> None:
    def create(**kwargs):
        raise error

    chat_model = make_openai_chat_model(create)
    messages = [{"role": "user", "content": "Hi"}]

    with pytest.raises(ChatModelError, match=expected):
        chat_model.generate_answer(messages, model="fake-model")

    with pytest.raises(ChatModelError, match=expected):
        list(chat_model.stream_answer(messages, model="fake-model"))


def test_describe_api_error_handles_timeouts() -> None:
    timeout = openai.APITimeoutError(request=httpx.Request("POST", "https://example.com"))

    assert "did not respond in time" in describe_api_error(timeout)
