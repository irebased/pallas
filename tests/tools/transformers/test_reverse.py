import pytest
from pallas.tools.transformers.reverse import Reverse
from pallas.tools.Tool import ToolError

@pytest.fixture
def reverse_tool():
    """Create a Reverse tool instance for testing."""
    return Reverse()

def test_reverse_basic(reverse_tool):
    """Test basic string reversal."""
    result, sep, error = reverse_tool.run("hello")
    assert error is None
    assert result == "olleh"
    assert sep is None

def test_reverse_empty_string(reverse_tool):
    """Test reversing an empty string."""
    result, sep, error = reverse_tool.run("")
    assert error is None
    assert result == ""
    assert sep is None

def test_reverse_single_char(reverse_tool):
    """Test reversing a single character."""
    result, sep, error = reverse_tool.run("a")
    assert error is None
    assert result == "a"
    assert sep is None

def test_reverse_palindrome(reverse_tool):
    """Test reversing a palindrome."""
    result, sep, error = reverse_tool.run("radar")
    assert error is None
    assert result == "radar"
    assert sep is None

def test_reverse_with_spaces(reverse_tool):
    """Test reversing a string with spaces."""
    result, sep, error = reverse_tool.run("hello world")
    assert error is None
    assert result == "dlrow olleh"
    assert sep is None

def test_reverse_with_special_chars(reverse_tool):
    """Test reversing a string with special characters."""
    result, sep, error = reverse_tool.run("hello!@#$%")
    assert error is None
    assert result == "%$#@!olleh"
    assert sep is None

def test_reverse_with_numbers(reverse_tool):
    """Test reversing a string with numbers."""
    result, sep, error = reverse_tool.run("12345")
    assert error is None
    assert result == "54321"
    assert sep is None

def test_reverse_with_mixed_content(reverse_tool):
    """Test reversing a string with mixed content."""
    result, sep, error = reverse_tool.run("Hello123!@#")
    assert error is None
    assert result == "#@!321olleH"
    assert sep is None

def test_reverse_with_separator(reverse_tool):
    """Test that the separator parameter is ignored and returned as is."""
    result, sep, error = reverse_tool.run("hello", input_separator=" ")
    assert error is None
    assert result == "olleh"
    assert sep == " "

def test_reverse_double_reverse(reverse_tool):
    """Test that reversing twice returns to the original string."""
    original = "Hello World!"
    result1, sep1, error1 = reverse_tool.run(original)
    assert error1 is None
    assert sep1 is None
    result2, sep2, error2 = reverse_tool.run(result1)
    assert error2 is None
    assert result2 == original
    assert sep2 is None