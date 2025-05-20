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
            'memoized': 0,
            'unbalanced_encodings': 0,
            'non_alternating': 0
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

def test_strict_alternating_chain_generation(mock_tools):
    """Test that chains follow strict alternating rules when enabled."""
    with patch("pallas.toolchain.ToolChainer.ToolDiscovery") as mock_discovery:
        mock_discovery.return_value.discover_tools.return_value = mock_tools
        chainer = ToolChainer(max_tree_size=3, verbose=True, strict_alternating=True)
        chainer._load_tools()

        # Generate chains
        available_tools = set(range(len(mock_tools)))
        chainer._generate_chains([], available_tools)

        # Verify all chains follow alternating pattern
        for chain in chainer.valid_chains:
            for i in range(len(chain) - 1):
                current_tool = mock_tools[chain[i]]
                next_tool = mock_tools[chain[i + 1]]

                # Check that encoders are followed by decoders and vice versa
                if '_encoder' in current_tool.name:
                    assert '_decoder' in next_tool.name, f"Encoder {current_tool.name} followed by non-decoder {next_tool.name}"
                elif '_decoder' in current_tool.name:
                    assert '_encoder' in next_tool.name, f"Decoder {current_tool.name} followed by non-encoder {next_tool.name}"

def test_balanced_chain_generation(mock_tools):
    """Test that chains are balanced when balance_encodings is enabled."""
    with patch("pallas.toolchain.ToolChainer.ToolDiscovery") as mock_discovery:
        mock_discovery.return_value.discover_tools.return_value = mock_tools
        chainer = ToolChainer(max_tree_size=3, verbose=True, balance_encodings=True)
        chainer._load_tools()

        # Generate chains
        available_tools = set(range(len(mock_tools)))
        chainer._generate_chains([], available_tools)

        # Verify we have some valid chains
        assert len(chainer.valid_chains) > 0, "No valid chains were generated"

        # Print chains for debugging
        if chainer.verbose:
            print("\nGenerated chains:")
            for chain in chainer.valid_chains:
                print(" -> ".join(mock_tools[i].name for i in chain))

        # Verify all chains are balanced
        for chain in chainer.valid_chains:
            encode_count = sum(1 for i in chain if '_encoder' in mock_tools[i].name)
            decode_count = sum(1 for i in chain if '_decoder' in mock_tools[i].name)

            # For even length chains, counts should be equal
            if len(chain) % 2 == 0:
                assert encode_count == decode_count, f"Unbalanced chain: {encode_count} encoders, {decode_count} decoders in {' -> '.join(mock_tools[i].name for i in chain)}"
            # For odd length chains, difference should be at most 1
            else:
                assert abs(encode_count - decode_count) <= 1, f"Unbalanced chain: {encode_count} encoders, {decode_count} decoders in {' -> '.join(mock_tools[i].name for i in chain)}"

def test_combined_alternating_and_balanced(mock_tools):
    """Test that chains follow both alternating and balancing rules when both are enabled."""
    with patch("pallas.toolchain.ToolChainer.ToolDiscovery") as mock_discovery:
        mock_discovery.return_value.discover_tools.return_value = mock_tools
        chainer = ToolChainer(max_tree_size=4, verbose=True,
                            strict_alternating=True, balance_encodings=True)
        chainer._load_tools()

        # Generate chains
        available_tools = set(range(len(mock_tools)))
        chainer._generate_chains([], available_tools)

        # Verify all chains follow both rules
        for chain in chainer.valid_chains:
            # Check alternating pattern
            for i in range(len(chain) - 1):
                current_tool = mock_tools[chain[i]]
                next_tool = mock_tools[chain[i + 1]]
                if '_encoder' in current_tool.name:
                    assert '_decoder' in next_tool.name
                elif '_decoder' in current_tool.name:
                    assert '_encoder' in next_tool.name

            # Check balancing
            encode_count = sum(1 for i in chain if '_encoder' in mock_tools[i].name)
            decode_count = sum(1 for i in chain if '_decoder' in mock_tools[i].name)
            if len(chain) % 2 == 0:
                assert encode_count == decode_count
            else:
                assert abs(encode_count - decode_count) <= 1