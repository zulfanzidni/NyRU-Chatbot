import os
from dotenv import load_dotenv

load_dotenv()


def get_llm_config() -> dict[str, str]:
    """Retrieve LLM configuration from environment variables.

    Returns:
        dict[str, str]: Dictionary containing 'base_url', 'api_key', and 'model'.
    """
    base_url = (os.getenv("OPENAI_BASE_URL") or "https://kenari.id/v1").rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY") or ""
    model = os.getenv("OPENAI_MODEL") or "meta-llama/Llama-3-8b"

    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
    }
