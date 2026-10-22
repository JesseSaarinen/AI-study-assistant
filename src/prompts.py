import re

from src.models import ChatTurn, RetrievedChunk

ANSWER_NOT_FOUND = "I could not find this in the uploaded notes."

# Earlier answers are shortened in the rewrite prompt. The rewrite only needs
# to know what was discussed, not every detail of the answer.
HISTORY_ANSWER_MAX_CHARS = 400

CITATION_MARKER_PATTERN = re.compile(r"\s*\[\s*Sources?\s+[^\]]*\]", re.IGNORECASE)

# The "Sources: [Source 1], [Source 2]" line that is appended when the model
# forgets to cite.
APPENDED_SOURCES_PATTERN = re.compile(r"\s*Sources:(\s*\[Source \d+\],?)+\s*$")


SYSTEM_PROMPT = f"""
You are an AI study assistant for university lecture notes.

You must answer questions using ONLY the provided context from the uploaded notes.

Rules:
1. Do not use external knowledge.
2. Do not guess.
3. If the answer is not clearly present in the provided context, say exactly:
   "{ANSWER_NOT_FOUND}"
4. If you answer, cite the relevant source using source markers like [Source 1].
5. If a sentence is supported by multiple sources, use separate adjacent markers:
   Correct: [Source 1][Source 2]
   Avoid: [Source 1, Source 2]
6. Put source markers directly after the sentence or paragraph they support.
7. Do not cite sources that do not support the answer.
8. Keep answers clear, concise, and suitable for a university student.
""".strip()


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """
    Build a context block from retrieved chunks.

    Each chunk is labelled as a source so the model can cite it in the answer.
    """

    if not chunks:
        return ""

    context_parts: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"""
[Source {index}]
File: {chunk.file_name}
Page: {chunk.page_number}
Text:
{chunk.text}
""".strip()
        )

    return "\n\n---\n\n".join(context_parts)


def build_rag_messages(
    question: str,
    chunks: list[RetrievedChunk],
) -> list[dict[str, str]]:
    """
    Build chat messages for the OpenAI-compatible chat model.
    """

    context_block = build_context_block(chunks)

    user_prompt = f"""
Question:
{question}

Context from uploaded notes:
{context_block}

Answering instructions:
- Answer only using the context above.
- If the context does not contain the answer, say exactly:
  "{ANSWER_NOT_FOUND}"
- If you answer, include source markers such as [Source 1] or [Source 2].
- If multiple sources support the same sentence, use separate adjacent markers:
  Correct: [Source 1][Source 2]
  Avoid: [Source 1, Source 2]
- Place source markers after the claims they support.
- Do not mention sources that are irrelevant.
""".strip()

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


REWRITE_SYSTEM_PROMPT = """
You rewrite follow-up questions so they can be used to search lecture notes.

Given a conversation between a student and a study assistant, and the student's
latest question, write one standalone question that a search engine could
answer without seeing the conversation.

Rules:
1. Replace words such as "it", "that" or "the second one" with the topic they refer to.
2. If the latest question is already standalone, return it unchanged.
3. Keep the language of the latest question.
4. Do not answer the question.
5. Return only the rewritten question, with no quotes, labels or explanation.
""".strip()


def _shorten(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.split())

    if len(cleaned) <= max_chars:
        return cleaned

    return cleaned[: max_chars - 3].rstrip() + "..."


def strip_citation_markers(text: str) -> str:
    """
    Remove [Source N] markers and an appended "Sources:" line from an answer.
    """

    without_appended_line = APPENDED_SOURCES_PATTERN.sub("", text)

    return CITATION_MARKER_PATTERN.sub("", without_appended_line).strip()


def build_conversation_transcript(history: list[ChatTurn]) -> str:
    """
    Format earlier turns as a compact transcript for the rewrite prompt.

    Citation markers are removed because [Source N] labels only make sense
    next to the sources of the turn they came from.
    """

    lines: list[str] = []

    for turn in history:
        answer = strip_citation_markers(turn.answer.answer)
        lines.append(f"Student: {_shorten(turn.question, HISTORY_ANSWER_MAX_CHARS)}")
        lines.append(f"Assistant: {_shorten(answer, HISTORY_ANSWER_MAX_CHARS)}")

    return "\n".join(lines)


def build_rewrite_messages(
    question: str,
    history: list[ChatTurn],
) -> list[dict[str, str]]:
    """
    Build chat messages that turn a follow-up question into a standalone one.
    """

    user_prompt = f"""
Conversation so far:
{build_conversation_transcript(history)}

Latest question:
{question}

Standalone question:
""".strip()

    return [
        {
            "role": "system",
            "content": REWRITE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
