from src.models import RetrievedChunk


ANSWER_NOT_FOUND = "I could not find this in the uploaded notes."


SYSTEM_PROMPT = f"""
You are an AI study assistant for university lecture notes.

You must answer questions using ONLY the provided context from the uploaded notes.

Rules:
1. Do not use external knowledge.
2. Do not guess.
3. If the answer is not clearly present in the provided context, say exactly:
   "{ANSWER_NOT_FOUND}"
4. If you answer, cite the relevant source using source markers like [Source 1].
5. Put source markers directly after the sentence or paragraph they support.
6. Do not cite sources that do not support the answer.
7. Keep answers clear, concise, and suitable for a university student.
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
