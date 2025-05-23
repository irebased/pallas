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

    def __str__(self):
        return f"ChainContext(current_chain={self.current_chain}, next_tool={self.next_tool}, target_length={self.target_length}, tools={self.tools})"

    def print_chain(self):
        return " -> ".join([self.tools[i].name for i in self.current_chain])

    def print_chain_with_next_tool(self):
        return " -> ".join([self.tools[i].name for i in self.current_chain] + [self.tools[self.next_tool].name])

    def print_next_tool(self):
        return self.tools[self.next_tool].name