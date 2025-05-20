import pytest
from unittest.mock import patch, mock_open
from pallas.toolchain.ToolChainer import ToolChainer
from pallas.tools.Tool import Tool

class MockTool(Tool):
    def __init__(self, name: str, domain_chars: set, range_chars: set):
        super().__init__(
            name=name,
            description=f"Mock tool {name}",
            domain_chars=domain_chars,
            range_chars=range_chars
        )

    def _process(self, input_str: str) -> str:
        return input_str

@pytest.fixture
def mock_tools():
    """Create a set of mock tools for testing."""
    return [
        MockTool("binary_encoder", {'0', '1'}, {'0', '1'}),
        MockTool("binary_decoder", {'0', '1'}, {'0', '1'}),
        MockTool("octal_encoder", {'0', '1', '2', '3', '4', '5', '6', '7'}, {'0', '1', '2', '3', '4', '5', '6', '7'}),
        MockTool("octal_decoder", {'0', '1', '2', '3', '4', '5', '6', '7'}, {'0', '1', '2', '3', '4', '5', '6', '7'}),
        MockTool("hex_encoder", {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'},
                {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'}),
        MockTool("hex_decoder", {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'},
                {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'})
    ]

def test_tool_chainer_initialization(mock_tools):
    """Test ToolChainer initialization and basic properties."""
    chainer = ToolChainer(chain_length=3, tools=mock_tools, verbose=True)
    assert chainer.chain_length == 3
    assert len(chainer.tools) == len(mock_tools)
    assert chainer.verbose is True
    assert chainer.invalid_connections == {}
    assert chainer.valid_chains == []
    assert chainer.pruned_chains == []
    assert chainer.visited_nodes == 0
    assert chainer.pruning_stats == {
        'redundant_pairs': 0,
        'char_set_mismatch': 0,
        'memoized': 0
    }

def test_calculate_max_tree_size(mock_tools):
    """Test the calculation of maximum tree size."""
    chainer = ToolChainer(chain_length=3, tools=mock_tools)
    max_size = chainer._calculate_max_tree_size()
    # For 6 tools and chain length 3, should be 6 + 6² + 6³ = 6 + 36 + 216 = 258
    assert max_size == 258

def test_redundant_encode_decode(mock_tools):
    """Test detection of redundant encode-decode pairs."""
    chainer = ToolChainer(chain_length=3, tools=mock_tools)

    # Test binary encode-decode pair
    assert chainer._is_redundant_encode_decode(mock_tools[0], mock_tools[1])  # binary_encoder -> binary_decoder
    assert chainer._is_redundant_encode_decode(mock_tools[1], mock_tools[0])  # binary_decoder -> binary_encoder

    # Test non-redundant pairs
    assert not chainer._is_redundant_encode_decode(mock_tools[0], mock_tools[2])  # binary_encoder -> octal_encoder
    assert chainer._is_redundant_encode_decode(mock_tools[2], mock_tools[3])  # octal_encoder -> octal_decoder (should be redundant)

def test_valid_connection(mock_tools):
    """Test connection validation logic."""
    chainer = ToolChainer(chain_length=3, tools=mock_tools)

    # Test valid connections
    assert chainer._is_valid_connection(mock_tools[0], mock_tools[2])  # binary -> octal (subset)
    assert chainer._is_valid_connection(mock_tools[2], mock_tools[0])  # octal -> binary (superset)

    # Test invalid connections (no overlap)
    no_overlap_tool = MockTool("no_overlap", {'x', 'y', 'z'}, {'x', 'y', 'z'})
    assert not chainer._is_valid_connection(mock_tools[0], no_overlap_tool)  # binary -> no_overlap (no overlap)

    # Test memoization
    assert mock_tools[0].name in chainer.invalid_connections
    assert no_overlap_tool.name in chainer.invalid_connections[mock_tools[0].name]
    assert chainer.pruning_stats['char_set_mismatch'] > 0

def test_dfs_with_pruning(mock_tools):
    """Test DFS with pruning strategies."""
    chainer = ToolChainer(chain_length=3, tools=mock_tools, verbose=True)

    # Start DFS
    chainer._dfs([], [])

    # Verify that some chains were found
    assert len(chainer.valid_chains) > 0

    # Verify that pruning occurred
    assert chainer.pruning_stats['redundant_pairs'] > 0 or chainer.pruning_stats['char_set_mismatch'] > 0
    assert len(chainer.pruned_chains) > 0

    # Verify visited nodes count
    assert chainer.visited_nodes > 0

def test_generate_chains_output(mock_tools):
    """Test chain generation and output file writing."""
    chainer = ToolChainer(chain_length=2, tools=mock_tools, verbose=True)

    # Mock file operations
    mock_file = mock_open()
    with patch('builtins.open', mock_file):
        chainer.generate_chains('test_output.txt')

    # Verify file was opened for writing
    mock_file.assert_called_once_with('test_output.txt', 'w')

    # Verify that some chains were written
    assert len(chainer.valid_chains) > 0

    # Verify timing information
    assert 'dfs' in chainer.phase_times

def test_empty_tools_list():
    """Test behavior with empty tools list."""
    chainer = ToolChainer(chain_length=3, tools=[], verbose=True)
    assert chainer._calculate_max_tree_size() == 0

    mock_file = mock_open()
    with patch('builtins.open', mock_file):
        chainer.generate_chains('test_output.txt')

    assert len(chainer.valid_chains) == 0
    assert chainer.visited_nodes == 1  # Root node is still visited