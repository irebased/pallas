class ToolError(Exception):
    """
    This is a generic exception that can be used to raise errors in tools.
    It provides a basic structure for errors, including tool name and message.
    """

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"{tool_name}: {message}")