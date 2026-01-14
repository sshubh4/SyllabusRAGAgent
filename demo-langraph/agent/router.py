import os
from anthropic import Anthropic
from agent.prompts import ROUTER_PROMPT
from dotenv import load_dotenv

load_dotenv()

_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        _client = Anthropic(api_key=api_key)
    return _client

def route_query(query: str) -> str:
    try:
        client = get_client()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            temperature=0,
            messages=[
                {"role": "user", "content": ROUTER_PROMPT.format(query=query)}
            ],
        )
        route = message.content[0].text.strip().lower()
        valid_routes = ["syllabus_rag", "weather", "chat"]
        return route if route in valid_routes else "chat"
    except Exception as e:
        return "chat"

