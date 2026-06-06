import json

from langgraph.graph import StateGraph

from agent.client import get_client
from agent.prompts import CHAT_PROMPT, SYLLABUS_RAG_PROMPT, WEATHER_PROMPT
from agent.rag import retrieve_with_meta
from agent.router import route_query
from agent.state import AgentState
from agent.tools import get_weather

_MODEL = "claude-sonnet-4-6"


def _claude(prompt: str, max_tokens: int = 800, temperature: float = 0.3) -> str:
    try:
        msg = get_client().messages.create(
            model=_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as exc:
        return f"Error calling Claude API: {exc}"


def router_node(state: AgentState) -> dict:
    return {"tool": route_query(state["query"])}


def syllabus_rag_node(state: AgentState) -> dict:
    try:
        context, citations = retrieve_with_meta(state["query"])
        result = _claude(
            SYLLABUS_RAG_PROMPT.format(context=context, query=state["query"]),
            max_tokens=2000,   # syllabi answers can be long — don't cut off
        )
        return {"result": result, "citations": citations}
    except Exception as exc:
        return {"result": f"Error accessing syllabus: {exc}", "citations": []}


def weather_node(state: AgentState) -> dict:
    extraction_prompt = f"""Extract city and date from this weather query.

Query: {state["query"]}

Return ONLY a JSON object with:
- "city": city name string
- "date": YYYY-MM-DD, "today", "tomorrow", "N days", or null

Examples:
- "What's the weather in New York?" → {{"city": "New York", "date": null}}
- "Weather in Paris tomorrow"       → {{"city": "Paris", "date": "tomorrow"}}

Return ONLY the JSON:"""

    try:
        raw = _claude(extraction_prompt, max_tokens=60, temperature=0)
        start, end = raw.find("{"), raw.rfind("}")
        extracted = json.loads(raw[start: end + 1])
        city = extracted.get("city", "").strip()
        if not city:
            return {"result": "Please include a city name in your weather question."}
        weather = get_weather(city, date=extracted.get("date") or None)
        return {"result": _claude(WEATHER_PROMPT.format(weather=weather, query=state["query"]))}
    except json.JSONDecodeError as exc:
        return {"result": f"Could not parse weather query: {exc}"}
    except Exception as exc:
        return {"result": f"Error fetching weather: {exc}"}


def chat_node(state: AgentState) -> dict:
    history = "\n".join(state.get("messages", [])) or "No previous conversation."
    return {"result": _claude(CHAT_PROMPT.format(history=history, query=state["query"]))}


def _route_decision(state: AgentState) -> str:
    tool = state.get("tool", "chat")
    return tool if tool in ("syllabus_rag", "weather", "chat") else "chat"


graph = StateGraph(AgentState)
graph.add_node("router", router_node)
graph.add_node("syllabus_rag", syllabus_rag_node)
graph.add_node("weather", weather_node)
graph.add_node("chat", chat_node)

graph.add_conditional_edges(
    "router",
    _route_decision,
    {"syllabus_rag": "syllabus_rag", "weather": "weather", "chat": "chat"},
)
graph.set_entry_point("router")

app = graph.compile()
