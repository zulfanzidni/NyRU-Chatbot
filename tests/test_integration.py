import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import config
import rag


def test_app_imports_cleanly():
    """Verify that config and rag modules have all expected symbols consumed by the app."""
    assert hasattr(rag, "build_chat_messages")
    assert hasattr(rag, "search_document")
    assert hasattr(rag, "stream_chat_response")
    assert hasattr(rag, "get_openai_client")
    assert hasattr(config, "get_llm_config")


def test_app_ast_has_no_legacy_gemini():
    """Verify that app.py does not import or reference google.generativeai or genai."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    source = app_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "google" not in alias.name
                assert "generativeai" not in alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert "google" not in node.module
                assert "generativeai" not in node.module

    assert "google.generativeai" not in source
    assert "genai" not in source


def test_app_ast_imports_required_rag_and_config():
    """Verify that app.py imports the new interfaces from rag and config."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    source = app_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.name)

    assert "get_llm_config" in imported_names
    assert "get_openai_client" in imported_names
    assert "build_chat_messages" in imported_names
    assert "search_document" in imported_names
    assert "stream_chat_response" in imported_names


def test_full_rag_pipeline_flow():
    """Integration test verifying end-to-end flow from search to prompt formatting to streaming."""
    # 1. Mock DB and embedder
    mock_db = MagicMock()
    mock_cursor = MagicMock()
    mock_db.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [
        ("Kopi susu gula aren Rp 18.000", 0.12),
        ("Buka setiap hari 09.00 - 22.00 WIB", 0.25),
    ]

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

    # 2. Search
    query = "Harga kopi susu berapa?"
    search_results = rag.search_document(mock_db, mock_embedder, query)
    context_docs = [doc["text"] for doc in search_results]

    assert len(context_docs) == 2
    assert "Kopi susu gula aren Rp 18.000" in context_docs[0]

    # 3. Build messages with history
    mock_history = [
        {"role": "user", "content": "Halo"},
        {"role": "assistant", "content": "Halo! Ada yang bisa dibantu?"},
    ]
    messages = rag.build_chat_messages(context_docs, query, history=mock_history)

    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert "Kopi susu gula aren" in messages[0]["content"]
    assert messages[1] == mock_history[0]
    assert messages[2] == mock_history[1]
    assert messages[3] == {"role": "user", "content": query}

    # 4. Stream response
    mock_client = MagicMock()
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(content="Harga kopi susu "))]
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock(delta=MagicMock(content="adalah Rp 18.000."))]
    mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])

    chunks = list(rag.stream_chat_response(mock_client, "meta-llama/Llama-3-8b", messages))
    full_response = "".join(chunks)

    assert full_response == "Harga kopi susu adalah Rp 18.000."


def test_app_ast_has_stream_error_handling():
    """Verify that app.py wraps stream_chat_response in a try-except block and handles errors."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    source = app_path.read_text(encoding="utf-8")
    assert "Gagal menghubungi layanan AI:" in source
    assert "messages.pop()" in source

    tree = ast.parse(source)
    found_try_with_stream = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and getattr(child.func, "id", None) == "stream_chat_response":
                    found_try_with_stream = True
                    break
    assert found_try_with_stream, "stream_chat_response must be wrapped in a try block"

