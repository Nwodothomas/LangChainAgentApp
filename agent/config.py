# agent/config.py
import os
from typing import Optional

try:
    # Optional: load from .env if python-dotenv is installed
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # It's okay if dotenv isn't installed
    pass


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Safe env getter that trims whitespace."""
    val = os.getenv(key, default)
    return val.strip() if isinstance(val, str) else val


def get_openai_api_key(raise_if_missing: bool = True) -> Optional[str]:
    """Return the OpenAI API key from env (OPENAI_API_KEY)."""
    api_key = get_env("OPENAI_API_KEY")
    if not api_key and raise_if_missing:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file."
        )
    return api_key


# ---- Backward compatibility export ----
# Some modules might still do: from agent.config import OPENAI_API_KEY
OPENAI_API_KEY = get_openai_api_key(raise_if_missing=False)

# Optional common config values
OPENAI_MODEL = get_env("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(get_env("OPENAI_TEMPERATURE", "0.2") or 0.2)
