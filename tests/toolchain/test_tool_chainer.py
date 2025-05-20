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
    with patch("pallas.toolchain.ToolChainer.ToolDiscovery") as mock_discovery:
        mock_discovery.return_value.discover_tools.return_value = mock_tools
        chainer = ToolChainer(max_tree_size=3, verbose=True)
        chainer._load_tools()  # Load the mock tools
        assert chainer.max_tree_size == 3
        assert chainer.tools == mock_tools
        assert chainer.verbose is True
        assert chainer.invalid_connections == {}
        assert chainer.valid_chains == []
        assert chainer.visited_nodes == 0
        assert chainer.pruning_stats == {
            'redundant_pairs': 0,
            'char_set_mismatch': 0,
            'memoized': 0
        }

def test_calculate_max_tree_size(mock_tools):
    """Test the calculation of maximum tree size."""
    with patch("pallas.toolchain.ToolChainer.ToolDiscovery") as mock_discovery:
        mock_discovery.return_value.discover_tools.return_value = mock_tools
        chainer = ToolChainer(max_tree_size=3)
        chainer._load_tools()  # Load the mock tools
        max_size = chainer._calculate_max_tree_size()
        assert max_size > 0
        assert max_size >= len(mock_tools)  # At least one level deep

def test_redundant_encode_decode(mock_tools):
    """Test detection of redundant encode-decode pairs."""
    with patch("pallas.toolchain.ToolChainer.ToolDiscovery") as mock_discovery:
        mock_discovery.return_value.discover_tools.return_value = mock_tools
        chainer = ToolChainer(max_tree_size=3)
        chainer._load_tools()  # Load the mock tools
        # Test encode -> decode
        assert chainer._is_redundant_encode_decode(mock_tools[0], mock_tools[1])  # binary_encoder -> binary_decoder
        # Test decode -> encode
        assert chainer._is_redundant_encode_decode(mock_tools[1], mock_tools[0])  # binary_decoder -> binary_encoder
        # Test non-redundant
        assert not chainer._is_redundant_encode_decode(mock_tools[0], mock_tools[2])  # binary_encoder -> octal_encoder
        # Test another encode -> decode pair
        assert chainer._is_redundant_encode_decode(mock_tools[2], mock_tools[3])  # octal_encoder -> octal_decoder
        # Test another decode -> encode pair
        assert chainer._is_redundant_encode_decode(mock_tools[3], mock_tools[2])  # octal_decoder -> octal_encoder

def test_valid_connection(mock_tools):
    """Test connection validation logic."""
    with patch("pallas.toolchain.ToolChainer.ToolDiscovery") as mock_discovery:
        mock_discovery.return_value.discover_tools.return_value = mock_tools
        chainer = ToolChainer(max_tree_size=3)
        chainer._load_tools()  # Load the mock tools
        # Test valid connection
        assert chainer._is_valid_connection(mock_tools[0], mock_tools[2])  # binary -> octal (subset)
        # Test invalid connection (should be pruned)
        no_overlap_tool = MockTool("no_overlap", {'x', 'y', 'z'}, {'x', 'y', 'z'})
        assert not chainer._is_valid_connection(mock_tools[0], no_overlap_tool)  # binary -> no_overlap (no overlap)

def test_dfs_with_pruning(mock_tools):
    """Test DFS with pruning strategies."""
    with patch("pallas.toolchain.ToolChainer.ToolDiscovery") as mock_discovery:
        mock_discovery.return_value.discover_tools.return_value = mock_tools
        chainer = ToolChainer(max_tree_size=3, verbose=True)
        chainer._load_tools()  # Load the mock tools
        chainer._generate_chains([], set(range(len(mock_tools))))
        assert len(chainer.valid_chains) > 0
        assert chainer.visited_nodes > 0
        # Verify no redundant pairs in chains
        for chain in chainer.valid_chains:
            for i in range(len(chain) - 1):
                assert not chainer._is_redundant_encode_decode(
                    mock_tools[chain[i]], mock_tools[chain[i + 1]]
                )

def test_generate_chains_output(mock_tools):
    """Test chain generation and output file writing."""
    with patch("pallas.toolchain.ToolChainer.ToolDiscovery") as mock_discovery:
        mock_discovery.return_value.discover_tools.return_value = mock_tools
        chainer = ToolChainer(max_tree_size=2, verbose=True)
        chainer._load_tools()  # Load the mock tools
        output_dir = chainer.generate_chains()
        assert output_dir.exists()
        output_file = output_dir / 'toolchain.txt'
        assert output_file.exists()
        with open(output_file) as f:
            content = f.read()
            assert len(content.strip().split('\n')) > 0

def test_empty_tools_list():
    """Test behavior with empty tools list."""
    with patch("pallas.toolchain.ToolChainer.ToolDiscovery") as mock_discovery:
        mock_discovery.return_value.discover_tools.return_value = []
        chainer = ToolChainer(max_tree_size=3, verbose=True)
        chainer._load_tools()  # Load the mock tools
        assert chainer._calculate_max_tree_size() == 0
        chainer._generate_chains([], set())
        assert len(chainer.valid_chains) == 0