from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    query: str
    messages: List[str]
    tool: Optional[str]
    context: Optional[str]
    result: Optional[str]
    citations: Optional[List[Dict[str, Any]]]           # [{source, page}] for chips
    retrieval_context: Optional[List[Dict[str, Any]]]   # [{text, source, page, score}] for viewer
    low_relevance: Optional[bool]                       # True if best-match distance > threshold
