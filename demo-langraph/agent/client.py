import os

import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

_client: Anthropic | None = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Add it to .env (local) or Streamlit secrets (cloud)."
            )
        _client = Anthropic(api_key=api_key)
    return _client
