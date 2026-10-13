"""RAG module for NYRU Coffee & Juice chatbot.

Handles prompt building, context formatting, and conversation memory.
"""

from typing import List, Optional, Dict, Any

SYSTEM_PROMPT_TEMPLATE = """Anda adalah asisten virtual ramah untuk NYRU COFFEE & JUICE di Bojonegoro.
Jawab pertanyaan pelanggan secara ramah, jelas, dan akurat berdasarkan Konteks Informasi di bawah ini.
Jika jawaban tidak ditemukan pada konteks, katakan dengan sopan bahwa Anda tidak mengetahui informasinya atau sarankan untuk menghubungi staf/WhatsApp kafe.

Konteks:
{context}
"""


def build_chat_messages(
    context_docs: Optional[List[str]],
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
    max_history_turns: int = 5,
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
