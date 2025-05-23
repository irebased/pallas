import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from pallas.toolrun.ToolRunner import ToolRunner
from pallas.tools.Tool import Tool, ToolError
from pallas.toolchain.ToolProvider import ToolProvider
from typing import Optional

class MockTool:
    def __init__(self, name, should_fail=False):
        self.name = name
        self.should_fail = should_fail

    def run(self, input_text, input_separator: Optional[str] = None):
        if self.should_fail:
            return "", input_separator, ToolError(self.name, "Mock error")
        return f"{input_text}_{self.name}", input_separator, None

@pytest.fixture
def mock_tools():
    return {
        'tool1': MockTool('tool1'),
        'tool2': MockTool('tool2'),
        'failing_tool': MockTool('failing_tool', should_fail=True)
    }

@pytest.fixture
def mock_tool_provider(mock_tools):
    discovery = ToolProvider()
    discovery.discover_tools = lambda: list(mock_tools.values())
    return discovery

@pytest.fixture
def toolchains_file(tmp_path):
    file_path = tmp_path / "toolchains.txt"
    with open(file_path, "w") as f:
        f.write("tool1 -> tool2\n")
        f.write("tool1 -> failing_tool\n")
        f.write("invalid_tool -> tool2\n")
    return str(file_path)

def test_tool_runner_initialization(toolchains_file, mock_tool_provider):
    """Test ToolRunner initialization with default parameters."""
    runner = ToolRunner(toolchains_file, "test_input", tool_provider=mock_tool_provider)
    assert runner.toolchains_file == Path(toolchains_file)
    assert runner.input_text == "test_input"
    assert runner.verbose is False
    assert runner.tools == {}
    assert isinstance(runner.run_id, str)
    assert runner.output_dir == Path('out')
    assert runner.stats == {
        'chains_processed': 0,
        'chains_succeeded': 0,
        'chains_failed': 0,
        'tools_loaded': 0
    }

def test_tool_runner_initialization_with_custom_params(toolchains_file, tmp_path, mock_tool_provider):
    """Test ToolRunner initialization with custom parameters."""
    runner = ToolRunner(
        toolchains_file,
        "test_input",
        tool_provider=mock_tool_provider,
        verbose=True,
        output_filename="custom_output.txt"
    )
    assert runner.verbose is True

def test_log_verbose(toolchains_file, mock_tool_provider):
    """Test logging in verbose mode."""
    runner = ToolRunner(toolchains_file, "test_input", tool_provider=mock_tool_provider, verbose=True)
    with patch('pallas.utils.logging_helpers.LoggingHelper.log') as mock_log:
        runner.logger.log("Test message")
        mock_log.assert_called_once_with("Test message")

def test_log_non_verbose(toolchains_file, mock_tool_provider):
    """Test logging in non-verbose mode."""
    runner = ToolRunner(toolchains_file, "test_input", tool_provider=mock_tool_provider, verbose=False)
    with patch('pallas.utils.logging_helpers.LoggingHelper.log') as mock_log:
        runner.logger.log("Test message")
        mock_log.assert_called_once_with("Test message")

def test_load_tools(toolchains_file, mock_tool_provider, mock_tools):
    """Test tool loading functionality."""
    runner = ToolRunner(toolchains_file, "test_input", tool_provider=mock_tool_provider, verbose=True)
    runner._load_tools()
    assert runner.tools == mock_tools
    assert runner.stats['tools_loaded'] == len(mock_tools)

def test_execute_chain_success(toolchains_file, mock_tool_provider, mock_tools):
    """Test successful chain execution."""
    runner = ToolRunner(toolchains_file, "test_input", tool_provider=mock_tool_provider)
    runner.tools = mock_tools
    output, error = runner._execute_chain(['tool1', 'tool2'])
    assert error is None
    assert output == "test_input_tool1_tool2"

def test_execute_chain_tool_not_found(toolchains_file, mock_tool_provider, mock_tools):
    """Test chain execution with non-existent tool."""
    runner = ToolRunner(toolchains_file, "test_input", tool_provider=mock_tool_provider)
    runner.tools = mock_tools
    output, error = runner._execute_chain(['nonexistent_tool'])
    assert isinstance(error, ToolError)
    assert error.tool_name == "nonexistent_tool"
    assert "Tool not found" in error.message
    assert output == ""

def test_execute_chain_tool_failure(toolchains_file, mock_tool_provider, mock_tools):
    """Test chain execution with failing tool."""
    runner = ToolRunner(toolchains_file, "test_input", tool_provider=mock_tool_provider)
    runner.tools = mock_tools
    output, error = runner._execute_chain(['failing_tool'])
    assert isinstance(error, ToolError)
    assert error.tool_name == "failing_tool"
    assert "Mock error" in error.message
    assert output == ""

def test_run_complete_execution(toolchains_file, mock_tool_provider, mock_tools, tmp_path):
    """Test complete execution of all chains."""
    runner = ToolRunner(toolchains_file, "test_input", tool_provider=mock_tool_provider, verbose=True)

    # Mock the output directory to be in tmp_path
    runner.output_dir = tmp_path / "out"
    runner.output_dir.mkdir()

    runner.run()

    # Verify output files were created
    success_files = list(runner.output_dir.glob('toolrun_succeeded_*.txt'))
    failed_files = list(runner.output_dir.glob('toolrun_failed_*.txt'))
    assert len(success_files) == 1
    assert len(failed_files) == 1

    # Verify content of success file
    success_content = success_files[0].read_text()
    assert "tool1 -> tool2" in success_content
    assert "test_input_tool1_tool2" in success_content

    # Verify content of failed file
    failed_content = failed_files[0].read_text()
    assert "failing_tool" in failed_content
    assert "invalid_tool" in failed_content

    # Verify stats
    assert runner.stats['chains_processed'] == 3
    assert runner.stats['chains_succeeded'] == 1
    assert runner.stats['chains_failed'] == 2
    assert runner.stats['tools_loaded'] == len(mock_tools)