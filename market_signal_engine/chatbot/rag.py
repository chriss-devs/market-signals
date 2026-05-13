"""RAG core: embed documents, retrieve relevant chunks, generate answers with Claude."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5

NO_INFO_RESPONSE = (
    "No encontré información sobre esto en los documentos disponibles."
)

SYSTEM_PROMPT = (
    "Eres un asistente que SOLO responde basándote en los fragmentos de documentos "
    "proporcionados. Si la información solicitada NO está en los fragmentos, responde "
    "exactamente: \"No encontré información sobre esto en los documentos disponibles.\" "
    "No uses conocimiento externo ni inventes información. Responde siempre en español."
)


@lru_cache(maxsize=1)
def _get_model() -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(text: str) -> list[float]:
    vec = _get_model().encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    vecs = _get_model().encode(texts, normalize_embeddings=True, batch_size=32)
    return [v.tolist() for v in vecs]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve(question: str, db_session) -> list[str]:
    """Return the top-K most relevant chunk texts for the given question."""
    from market_signal_engine.chatbot.db import get_all_chunks

    chunks = get_all_chunks(db_session)
    if not chunks:
        return []

    q_vec = embed(question)
    scored = []
    for chunk in chunks:
        if not chunk.embedding:
            continue
        c_vec = json.loads(chunk.embedding)
        score = _cosine(q_vec, c_vec)
        scored.append((score, chunk.content))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:TOP_K]]


def answer(question: str, db_session) -> str:
    """Generate a grounded answer using Claude, restricted to retrieved chunks."""
    from market_signal_engine.config.settings import settings

    chunks = retrieve(question, db_session)
    if not chunks:
        return NO_INFO_RESPONSE

    context = "\n\n---\n\n".join(chunks)
    user_msg = f"Fragmentos relevantes:\n{context}\n\nPregunta: {question}"

    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text
