import pytest
from pallas.tools.Tool import Tool
from pallas.tools.ToolError import ToolError

class MockTool(Tool):
    def _process(self, input_str: str) -> str:
        raise ValueError("Test error")

def test_tool_error_handling():
    tool = MockTool(
        name="test_tool",
        description="Test tool for error handling",
        domain_chars=set("abc"),
        range_chars=set("123")
    )

    # Test with error from previous tool
    prev_error = ToolError("prev_tool", "Previous error")
    result, error = tool.run("test", error=prev_error)
    assert error == prev_error
    assert result == "test"

    # Test with processing error
    result, error = tool.run("test")
    assert error is not None
    assert isinstance(error, ToolError)
    assert error.tool_name == "test_tool"
    assert "Test error" in str(error)
    assert result == "test"