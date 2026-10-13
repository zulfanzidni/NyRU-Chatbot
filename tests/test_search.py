from unittest.mock import MagicMock
import pytest
from rag import search_document


def test_search_document_executes_parameterized_query():
    mock_db = MagicMock()
    mock_cursor = MagicMock()
    mock_db.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [("Menu A", 0.12), ("Menu B", 0.35)]

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

    results = search_document(mock_db, mock_embedder, "menu kopi", k_top=5)

    assert len(results) == 2
    assert results[0] == {"text": "Menu A", "distance": 0.12}
    mock_cursor.execute.assert_called_once()
    sql_arg, params = mock_cursor.execute.call_args[0]
    assert "LIMIT %s" in sql_arg
    assert params[1] == 5


def test_search_document_handles_db_failure():
    mock_db = MagicMock()
    mock_db.cursor.side_effect = Exception("Connection lost")
    mock_db.ping.side_effect = Exception("Reconnect failed")
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1]

    results = search_document(mock_db, mock_embedder, "test")
    assert results == []


def test_search_document_reconnects_and_retries_on_failure():
    mock_db = MagicMock()
    cursor_fail = MagicMock()
    cursor_fail.execute.side_effect = Exception("Socket dropped")
    cursor_ok = MagicMock()
    cursor_ok.fetchall.return_value = [("Recovered item", 0.08)]
    mock_db.cursor.side_effect = [cursor_fail, cursor_ok]
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1]

    results = search_document(mock_db, mock_embedder, "test")
    assert len(results) == 1
    assert results[0] == {"text": "Recovered item", "distance": 0.08}
    mock_db.ping.assert_called_once_with(reconnect=True, attempts=2, delay=1)


def test_search_document_clamps_k_top_minimum_one():
    mock_db = MagicMock()
    mock_cursor = MagicMock()
    mock_db.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = []
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1]

    search_document(mock_db, mock_embedder, "test", k_top=0)
    _, params = mock_cursor.execute.call_args[0]
    assert params[1] == 1

    search_document(mock_db, mock_embedder, "test", k_top=-3)
    _, params = mock_cursor.execute.call_args[0]
    assert params[1] == 1


def test_search_document_default_k_top():
    mock_db = MagicMock()
    mock_cursor = MagicMock()
    mock_db.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = []
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1]

    search_document(mock_db, mock_embedder, "test")
    _, params = mock_cursor.execute.call_args[0]
    assert params[1] == 10


def test_search_document_handles_query_execution_error_and_closes_cursor():
    mock_db = MagicMock()
    mock_cursor = MagicMock()
    mock_db.cursor.return_value = mock_cursor
    mock_cursor.execute.side_effect = Exception("SQL execution error")
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1]

    results = search_document(mock_db, mock_embedder, "test")
    assert results == []
    # Closed once during initial failure and once during retry failure
    assert mock_cursor.close.call_count == 2


def test_search_document_closes_cursor_on_success():
    mock_db = MagicMock()
    mock_cursor = MagicMock()
    mock_db.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [("Sample", 0.05)]
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1]

    results = search_document(mock_db, mock_embedder, "test")
    assert len(results) == 1
    mock_cursor.close.assert_called_once()
