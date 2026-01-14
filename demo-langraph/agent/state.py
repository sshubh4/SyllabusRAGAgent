from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    query: str
    messages: List[str]
    tool: Optional[str]
    context: Optional[str]
    result: Optional[str]

