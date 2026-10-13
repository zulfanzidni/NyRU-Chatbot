import os
import pytest
from config import get_llm_config

def test_get_llm_config_reads_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://kenari.id/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("OPENAI_MODEL", "meta-llama/Llama-3-8b")
    
    cfg = get_llm_config()
    assert cfg["base_url"] == "https://kenari.id/v1"
    assert cfg["api_key"] == "test-key-123"
    assert cfg["model"] == "meta-llama/Llama-3-8b"

def test_get_llm_config_default_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    
    cfg = get_llm_config()
    assert cfg["base_url"] == "https://kenari.id/v1"
    assert cfg["model"] == "meta-llama/Llama-3-8b"

def test_get_llm_config_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://kenari.id/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    cfg = get_llm_config()
    assert cfg["base_url"] == "https://kenari.id/v1"

def test_get_llm_config_empty_env_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    
    cfg = get_llm_config()
    assert cfg["base_url"] == "https://kenari.id/v1"
    assert cfg["model"] == "meta-llama/Llama-3-8b"
    assert cfg["api_key"] == ""
