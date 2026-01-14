import os
import json
from langgraph.graph import StateGraph
from anthropic import Anthropic
from dotenv import load_dotenv
from agent.state import AgentState
from agent.router import route_query
from agent.rag import retrieve
from agent.tools import get_weather
from agent.prompts import (
    SYLLABUS_RAG_PROMPT,
    CHAT_PROMPT,
    WEATHER_PROMPT,
)

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

def claude_call(prompt: str) -> str:
    try:
        client = get_client()
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"Error calling Claude API: {str(e)}"

def router_node(state: AgentState):
    return {"tool": route_query(state["query"])}

def syllabus_rag_node(state: AgentState):
    try:
        context = retrieve(state["query"])
        prompt = SYLLABUS_RAG_PROMPT.format(
            context=context, query=state["query"]
        )
        return {"result": claude_call(prompt)}
    except Exception as e:
        error_msg = str(e)
        return {"result": f"I encountered an error accessing your syllabus: {error_msg}"}

def weather_node(state: AgentState):
    try:
        extraction_prompt = f"""Extract the city name and date from this weather query.
        
            Query: {state["query"]}

            Return ONLY a JSON object with "city" and "date" keys.
            - "city": The city name (e.g., "New York", "London")
            - "date": Optional. Date in YYYY-MM-DD format, or "today", "tomorrow", or relative like "2 days". If no date mentioned, use null.

            Example responses:
            - "What's the weather in New York?" → {{"city": "New York", "date": null}}
            - "Weather in Paris tomorrow" → {{"city": "Paris", "date": "tomorrow"}}
            - "What's the forecast for London on 2024-12-25?" → {{"city": "London", "date": "2024-12-25"}}
            - "Weather in Tokyo 3 days from now" → {{"city": "Tokyo", "date": "3 days"}}

            Return ONLY the JSON, nothing else:"""
        
        extraction_result = claude_call(extraction_prompt)

        extraction_result = extraction_result.strip()
        
        
        start_idx = extraction_result.find('{')
        end_idx = extraction_result.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = extraction_result[start_idx:end_idx + 1]
            try:
                extracted = json.loads(json_str)
            except json.JSONDecodeError:
                
                extracted = json.loads(extraction_result)
        else:
            
            extracted = json.loads(extraction_result)
        
        city = extracted.get("city", "").strip()
        date = extracted.get("date")
        
        if not city:
            return {"result": "I couldn't identify a city name in your query. Please specify a city"}
        
        # Get weather data
        weather = get_weather(city, date=date if date else None)
        prompt = WEATHER_PROMPT.format(
            weather=weather, query=state["query"]
        )
        return {"result": claude_call(prompt)}
    except json.JSONDecodeError as e:
        return {"result": f"Error parsing weather query: {str(e)}."}
    except Exception as e:
        return {"result": f"Error fetching weather: {str(e)}"}

def chat_node(state: AgentState):
    history_str = "\n".join(state.get("messages", [])) if state.get("messages") else "No previous conversation."
    prompt = CHAT_PROMPT.format(
        history=history_str, query=state["query"]
    )
    return {"result": claude_call(prompt)}

graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("syllabus_rag", syllabus_rag_node)
graph.add_node("weather", weather_node)
graph.add_node("chat", chat_node)

def route_decision(state: AgentState) -> str:
    tool = state.get("tool", "chat")
    return tool if tool in ["syllabus_rag", "weather", "chat"] else "chat"

graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "syllabus_rag": "syllabus_rag",
        "weather": "weather",
        "chat": "chat",
    },
)

graph.set_entry_point("router")

app = graph.compile()

