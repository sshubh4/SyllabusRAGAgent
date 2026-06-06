from agent.client import get_client
from agent.prompts import ROUTER_PROMPT


def route_query(query: str) -> str:
    try:
        message = get_client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": ROUTER_PROMPT.format(query=query)}],
        )
        route = message.content[0].text.strip().lower()
        return route if route in ("syllabus_rag", "weather", "chat") else "chat"
    except Exception:
        return "chat"
