from typing import List, Optional
from pallas.tools.Tool import Tool
from pallas.tools.tool_map import tools as tool_map

class ToolProvider:
    """Class responsible for providing tools."""

    def __init__(self, tool_names: Optional[List[str]] = None):
        """Initialize the tool discovery process.

        Args:
            tool_names: Optional list of tool names to load. If None, loads all available tools.
        """
        self.tool_names = tool_names

    def discover_tools(self) -> List[Tool]:
        """Discover and instantiate the specified tools.

        Returns:
            List[Tool]: List of tool instances.
        """
        if self.tool_names is None:
            # If no tools specified, use all available tools
            return [tool_class() for tool_class in tool_map.values()]

        # Otherwise, only instantiate the specified tools
        return [tool_map[name]() for name in self.tool_names if name in tool_map]