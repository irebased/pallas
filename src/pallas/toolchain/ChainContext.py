from dataclasses import dataclass
from typing import List
from pallas.tools.Tool import Tool

@dataclass
class ChainContext:
    """Context for the current step of tool chain traversal."""
    current_chain: List[str]
    next_tool: str
    target_length: int
    tools: List[Tool]