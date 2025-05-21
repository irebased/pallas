import pytest
from pallas.tools.Tool import Tool, ToolError
from typing import Optional

class MockTool(Tool):
    """A mock tool for testing."""

    name = "test_tool"
    description = "A test tool"
    domain_chars = "abc"
    range_chars = "xyz"

    def _process(self, input_str: str, input_separator: Optional[str] = None) -> tuple[str, Optional[str]]:
        """Process the input string."""
        if input_str == "abc":
            raise ValueError("Test error")
        return input_str

def test_tool_error_handling():
    # Test with error from previous tool
    prev_error = ToolError("prev_tool", "Previous error")
    result, sep, error = MockTool().run("abc", error=prev_error)
    assert error == prev_error
    assert result == "abc"

    # Test with processing error
    result, sep, error = MockTool().run("abc")
    assert error is not None
    assert isinstance(error, ToolError)
    assert "Test error" in str(error)