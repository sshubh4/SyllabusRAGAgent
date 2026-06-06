from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    query: str
    messages: List[str]
    tool: Optional[str]
    context: Optional[str]
    result: Optional[str]
    citations: Optional[List[Dict[str, Any]]]  # [{source, page}, ...]
