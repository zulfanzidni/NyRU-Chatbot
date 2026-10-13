import pytest
from rag import build_chat_messages


def test_build_chat_messages_basic():
    context_docs = ["Menu: Cireng Salju Rp 10000"]
    query = "Berapa harga cireng?"
    messages = build_chat_messages(context_docs, query)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "NYRU" in messages[0]["content"]
    assert "Cireng Salju" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == query


def test_build_chat_messages_with_history():
    context_docs = ["Jam buka 10.00 - 23.00 WIB"]
    history = [
        {"role": "user", "content": "Halo"},
        {"role": "assistant", "content": "Halo! Ada yang bisa dibantu?"},
    ]
    query = "Buka sampai jam berapa?"
    messages = build_chat_messages(context_docs, query, history=history)

    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Halo"}
    assert messages[2] == {
        "role": "assistant",
        "content": "Halo! Ada yang bisa dibantu?",
    }
    assert messages[3]["content"] == query


def test_build_chat_messages_prunes_long_history():
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(12)
    ]
    messages = build_chat_messages(
        [], "query", history=history, max_history_turns=4
    )
    # 1 system + 4 history + 1 current query = 6 messages
    assert len(messages) == 6
    assert messages[-1]["content"] == "query"
    # Ensure it keeps the latest 4 turns
    assert messages[1]["content"] == "msg 8"
    assert messages[4]["content"] == "msg 11"


def test_build_chat_messages_empty_context():
    messages = build_chat_messages([], "Apa menu favorit?")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Konteks Informasi:" in messages[0]["content"]


def test_build_chat_messages_zero_history_turns():
    history = [
        {"role": "user", "content": "Halo"},
        {"role": "assistant", "content": "Halo!"},
    ]
    messages = build_chat_messages(
        [], "Ada promo?", history=history, max_history_turns=0
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "Ada promo?"


def test_build_chat_messages_default_max_history_turns_is_six():
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(10)
    ]
    messages = build_chat_messages([], "query", history=history)
    # Default is 6 turns: 1 system + 6 history + 1 user query = 8 messages
    assert len(messages) == 8
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "msg 4"
    assert messages[6]["content"] == "msg 9"
    assert messages[7]["content"] == "query"

