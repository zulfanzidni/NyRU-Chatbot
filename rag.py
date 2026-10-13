"""RAG module for NYRU Coffee & Juice chatbot.

Handles prompt building, context formatting, and conversation memory.
"""

import json
import logging
from typing import Any, Dict, Generator, List, Optional

from openai import OpenAI

from config import get_llm_config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """Anda adalah asisten virtual ramah untuk NYRU COFFEE & JUICE di Bojonegoro.
Jawab pertanyaan pelanggan secara ramah, jelas, dan akurat berdasarkan Konteks Informasi di bawah ini.
Jika jawaban tidak ditemukan pada konteks, katakan dengan sopan bahwa Anda tidak mengetahui informasinya atau sarankan untuk menghubungi staf/WhatsApp kafe.

Konteks Informasi:
{context}
"""


def build_chat_messages(
    context_docs: Optional[List[str]],
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
    max_history_turns: int = 6,
) -> List[Dict[str, Any]]:
    """Build OpenAI-compatible chat messages with persona, context, and sliding history window.

    Args:
        context_docs: Chunks of context retrieved from the knowledge base.
        query: Current user query.
        history: Prior conversation turns formatted as OpenAI message dicts.
        max_history_turns: Maximum number of recent history messages to retain.

    Returns:
        List of message dictionaries ready for OpenAI chat completion.
    """
    context_str = "\n".join(context_docs or [])
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_str)

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    if history and max_history_turns > 0:
        messages.extend(history[-max_history_turns:])

    messages.append({"role": "user", "content": query})

    return messages


def search_document(
    database: Any,
    embedder: Any,
    query: str,
    k_top: int = 10,
) -> List[Dict[str, Any]]:
    """Search for relevant documents in TiDB vector store using cosine distance.

    Args:
        database: TiDB/MySQL database connection object.
        embedder: Embedding model instance with encode() method.
        query: User text query.
        k_top: Number of top documents to retrieve (minimum 1).

    Returns:
        List of dictionaries containing 'text' and 'distance'.
    """
    try:
        safe_k = max(1, int(k_top))
    except (ValueError, TypeError):
        safe_k = 10

    try:
        query_embedding_list = embedder.encode(query).tolist()
        query_embedding_str = json.dumps(query_embedding_list)
    except Exception as e:
        logger.error("Failed to generate query embedding: %s", e)
        return []

    sql_query = """
        SELECT text, vec_cosine_distance(embedding, %s) AS distance
        FROM documents
        ORDER BY distance ASC
        LIMIT %s;
    """

    def _execute_query() -> List[Dict[str, Any]]:
        curr = database.cursor()
        try:
            curr.execute(sql_query, (query_embedding_str, safe_k))
            return [{"text": row[0], "distance": row[1]} for row in curr.fetchall()]
        finally:
            curr.close()

    try:
        return _execute_query()
    except Exception as e:
        logger.warning("Database query failed (%s), attempting reconnect...", e)
        try:
            if hasattr(database, "ping"):
                database.ping(reconnect=True, attempts=2, delay=1)
            return _execute_query()
        except Exception as retry_e:
            logger.error("Database search failed after reconnect: %s", retry_e)
            return []


def get_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 30.0,
) -> OpenAI:
    """Initialize and return an OpenAI client instance.

    Args:
        api_key: OpenAI API key. If omitted or None, falls back to config.
        base_url: OpenAI base URL. If omitted or None, falls back to config.
        timeout: Request timeout in seconds. Defaults to 30.0.

    Returns:
        OpenAI: Configured OpenAI client.

    Raises:
        ValueError: If api_key is missing or empty.
    """
    if api_key is None or base_url is None:
        cfg = get_llm_config()
        if api_key is None:
            api_key = cfg.get("api_key")
        if base_url is None:
            base_url = cfg.get("base_url")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def stream_chat_response(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
) -> Generator[str, None, None]:
    """Stream chat response tokens from OpenAI chat completions API.

    Args:
        client: Configured OpenAI client instance.
        model: Model identifier.
        messages: List of message dictionaries formatted for OpenAI API.

    Yields:
        str: Text delta chunks as they arrive from the stream.
    """
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


