import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

_client: Anthropic | None = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Add it to your .env file.")
        _client = Anthropic(api_key=api_key)
    return _client
