from unittest.mock import MagicMock, patch
import pytest
from rag import stream_chat_response, get_openai_client


def test_stream_chat_response_yields_deltas():
    mock_client = MagicMock()
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Halo "

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = "kak!"

    chunk3 = MagicMock()
    chunk3.choices = [MagicMock()]
    chunk3.choices[0].delta.content = None

    mock_client.chat.completions.create.return_value = [chunk1, chunk2, chunk3]

    tokens = list(
        stream_chat_response(
            mock_client, "test-model", [{"role": "user", "content": "hi"}]
        )
    )
    assert tokens == ["Halo ", "kak!"]
    mock_client.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )


def test_stream_chat_response_handles_empty_choices():
    mock_client = MagicMock()
    chunk_empty = MagicMock()
    chunk_empty.choices = []

    mock_client.chat.completions.create.return_value = [chunk_empty]

    tokens = list(stream_chat_response(mock_client, "test-model", []))
    assert tokens == []


def test_get_openai_client_validation():
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not configured"):
        get_openai_client(api_key="", base_url="https://kenari.id/v1")


def test_get_openai_client_none_without_config_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not configured"):
        get_openai_client(api_key=None)


@patch("rag.OpenAI")
def test_get_openai_client_success_explicit(mock_openai_cls):
    client = get_openai_client(
        api_key="sk-test-key", base_url="https://kenari.id/v1"
    )
    mock_openai_cls.assert_called_once_with(
        api_key="sk-test-key", base_url="https://kenari.id/v1", timeout=30.0
    )
    assert client == mock_openai_cls.return_value


@patch("rag.OpenAI")
def test_get_openai_client_fallback_to_config(mock_openai_cls, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1/")

    client = get_openai_client()
    mock_openai_cls.assert_called_once_with(
        api_key="sk-env-key", base_url="https://api.example.com/v1", timeout=30.0
    )
    assert client == mock_openai_cls.return_value


@patch("rag.OpenAI")
def test_get_openai_client_custom_timeout(mock_openai_cls):
    client = get_openai_client(
        api_key="sk-test-key", base_url="https://kenari.id/v1", timeout=15.0
    )
    mock_openai_cls.assert_called_once_with(
        api_key="sk-test-key", base_url="https://kenari.id/v1", timeout=15.0
    )
    assert client == mock_openai_cls.return_value

