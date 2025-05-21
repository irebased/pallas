import pytest
from unittest.mock import patch, MagicMock
from pallas.toolchain.ToolChainer import ToolChainer
from pallas.toolchain.rules.RuleEnforcer import RuleEnforcer
from pallas.toolchain.rules.AlternatingRule import AlternatingRule
from pallas.toolchain.rules.BalancingRule import BalancingRule
from pallas.toolchain.rules.RedundantPairRule import RedundantPairRule
from pallas.toolchain.rules.CharacterSetRule import CharacterSetRule
from pallas.tools.Tool import Tool
from pallas.toolchain.ToolProvider import ToolProvider

class MockTool(Tool):
    def __init__(self, name, domain_chars, range_chars):
        super().__init__()
        self.name = name
        self.description = f"Mock tool {name}"
        self.domain_chars = domain_chars
        self.range_chars = range_chars

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

@pytest.fixture
def mock_tool_provider(mock_tools):
    discovery = ToolProvider()
    discovery.discover_tools = lambda: mock_tools
    return discovery

def test_tool_chainer_initialization(mock_tool_provider):
    """Test ToolChainer initialization and basic properties."""
    enforcer = RuleEnforcer([RedundantPairRule, CharacterSetRule])
    chainer = ToolChainer(tool_provider=mock_tool_provider, max_tree_size=3, rule_enforcer=enforcer, verbose=True)
    assert chainer.max_tree_size == 3
    assert chainer.verbose is True
    assert chainer.rule_enforcer == enforcer
    assert chainer.tools == []
    assert chainer.valid_chains == []
    assert chainer.visited_nodes == 0

def test_generate_chains_with_rules(mock_tool_provider):
    enforcer = RuleEnforcer([RedundantPairRule, CharacterSetRule])
    chainer = ToolChainer(tool_provider=mock_tool_provider, max_tree_size=3, rule_enforcer=enforcer, verbose=True)
    output_dir = chainer.generate_chains()
    assert output_dir.exists()
    output_file = output_dir / 'toolchain.txt'
    assert output_file.exists()
    with open(output_file) as f:
        content = f.read()
        assert len(content.strip().split('\n')) > 0

def test_generate_chains_with_alternating_and_balancing(mock_tool_provider):
    enforcer = RuleEnforcer([AlternatingRule, BalancingRule, RedundantPairRule, CharacterSetRule])
    chainer = ToolChainer(tool_provider=mock_tool_provider, max_tree_size=4, rule_enforcer=enforcer, verbose=True)
    output_dir = chainer.generate_chains()
    assert output_dir.exists()
    output_file = output_dir / 'toolchain.txt'
    assert output_file.exists()
    with open(output_file) as f:
        content = f.read()
        chains = content.strip().split('\n')
        assert len(chains) > 0
        for chain in chains:
            tools = chain.split(' -> ')
            assert len(tools) == 4

def test_generate_chains_output(mock_tool_provider):
    """Test chain generation and output file writing."""
    chainer = ToolChainer(tool_provider=mock_tool_provider, max_tree_size=2, verbose=True)
    chainer._load_tools()  # Load the mock tools
    output_dir = chainer.generate_chains()
    assert output_dir.exists()
    output_file = output_dir / 'toolchain.txt'
    assert output_file.exists()
    with open(output_file) as f:
        content = f.read()
        assert len(content.strip().split('\n')) > 0