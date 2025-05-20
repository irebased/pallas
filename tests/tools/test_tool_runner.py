import pytest
from pathlib import Path
from unittest.mock import mock_open, patch
from pallas.tools.Tool import Tool, ToolError
from pallas.toolrun.ToolRunner import ToolRunner

class MockTool(Tool):
    """Mock tool for testing."""
    def __init__(self, name: str, output: str = None, error: ToolError = None):
        super().__init__(name, "Test tool", set("abc"), set("xyz"))
        self._output = output or "output"
        self._error = error

    def _process(self, input_str: str) -> str:
        if self._error:
            raise self._error
        return self._output

@pytest.fixture
def mock_tools():
    """Create a list of mock tools for testing."""
    return [
        MockTool("tool1", "output1"),
        MockTool("tool2", "output2"),
        MockTool("tool3", "output3"),
        MockTool("error_tool", error=ToolError(tool_name="error_tool", message="Test error"))
    ]

@pytest.fixture
def toolchains_file(tmp_path):
    """Create a temporary toolchains file."""
    chains = [
        "tool1 -> tool2 -> tool3",
        "tool1 -> error_tool -> tool3",
        "nonexistent -> tool2",
        "# Comment line",
        "",
        "tool1 -> tool2"
    ]
    file_path = tmp_path / "toolchains.txt"
    with open(file_path, "w") as f:
        f.write("\n".join(chains))
    return file_path

def test_tool_runner_initialization(toolchains_file):
    """Test ToolRunner initialization."""
    runner = ToolRunner(toolchains_file, "test input")
    assert runner.toolchains_file == Path(toolchains_file)
    assert runner.input_text == "test input"
    assert runner.tools == {}
    assert runner.output_file.parent == Path("out")
    assert runner.output_file.name.startswith("toolrun_")
    assert runner.output_file.name.endswith(".txt")
    assert not runner.verbose

def test_tool_runner_verbose_initialization(toolchains_file):
    """Test ToolRunner initialization with verbose mode."""
    runner = ToolRunner(toolchains_file, "test input", verbose=True)
    assert runner.verbose

def test_load_tools(mock_tools):
    """Test tool loading functionality."""
    with patch("pallas.toolrun.ToolRunner.ToolDiscovery") as mock_discovery:
        mock_discovery.return_value.discover_tools.return_value = mock_tools
        runner = ToolRunner("dummy.txt", "test input")
        runner._load_tools()
        assert len(runner.tools) == 4
        assert "tool1" in runner.tools
        assert "error_tool" in runner.tools
        assert runner.stats["tools_loaded"] == 4

def test_execute_chain_success(mock_tools):
    """Test successful chain execution."""
    with patch("pallas.toolrun.ToolRunner.ToolDiscovery") as mock_discovery:
        mock_discovery.return_value.discover_tools.return_value = mock_tools
        runner = ToolRunner("dummy.txt", "test input")
        runner._load_tools()
        output, error = runner._execute_chain(["tool1", "tool2", "tool3"])
        assert output == "output3"
        assert error is None

def test_execute_chain_tool_not_found(mock_tools):
    """Test chain execution with non-existent tool."""
    with patch("pallas.toolrun.ToolRunner.ToolDiscovery") as mock_discovery:
        mock_discovery.return_value.discover_tools.return_value = mock_tools
        runner = ToolRunner("dummy.txt", "test input")
        runner._load_tools()
        output, error = runner._execute_chain(["nonexistent", "tool2"])
        assert output == ""
        assert isinstance(error, ToolError)
        assert "Tool not found" in str(error)

def test_execute_chain_tool_error(mock_tools):
    """Test chain execution with tool error."""
    with patch("pallas.toolrun.ToolRunner.ToolDiscovery") as mock_discovery:
        mock_discovery.return_value.discover_tools.return_value = mock_tools
        runner = ToolRunner("dummy.txt", "test input")
        runner._load_tools()
        output, error = runner._execute_chain(["tool1", "error_tool", "tool3"])
        assert output == ""
        assert isinstance(error, ToolError)
        assert "Test error" in str(error)

def test_run_with_custom_tools_dir(tmp_path, mock_tools):
    """Test running with custom tools directory."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()

    with patch("pallas.toolrun.ToolRunner.ToolDiscovery") as mock_discovery:
        mock_discovery.return_value.discover_tools.return_value = mock_tools
        runner = ToolRunner("dummy.txt", "test input", tools_dir=tools_dir)
        runner._load_tools()
        assert len(runner.tools) == 4