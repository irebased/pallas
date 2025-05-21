import pytest
from unittest.mock import patch
from pallas.toolchain.ToolChainer import ToolChainer
from pallas.toolchain.rules.RuleEnforcer import RuleEnforcer
from pallas.toolchain.rules.AlternatingRule import AlternatingRule
from pallas.toolchain.rules.BalancingRule import BalancingRule
from pallas.toolchain.rules.RedundantPairRule import RedundantPairRule
from pallas.toolchain.rules.CharacterSetRule import CharacterSetRule
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
    enforcer = RuleEnforcer([RedundantPairRule, CharacterSetRule])
    chainer = ToolChainer(max_tree_size=3, rule_enforcer=enforcer, verbose=True)
    chainer.tools = mock_tools
    assert chainer.max_tree_size == 3
    assert chainer.tools == mock_tools
    assert chainer.verbose is True
    assert chainer.valid_chains == []
    assert chainer.visited_nodes == 0

def test_generate_chains_with_rules(mock_tools):
    enforcer = RuleEnforcer([RedundantPairRule, CharacterSetRule])
    chainer = ToolChainer(max_tree_size=3, rule_enforcer=enforcer, verbose=True)
    chainer.tools = mock_tools
    chainer.generate_chains()
    # All chains should not have redundant pairs or char set mismatches
    for chain in chainer.valid_chains:
        for i in range(len(chain) - 1):
            t1 = chainer.tools[chain[i]]
            t2 = chainer.tools[chain[i+1]]
            # Redundant pair check
            base1 = t1.name.rsplit('_', 1)[0]
            base2 = t2.name.rsplit('_', 1)[0]
            op1 = t1.name.rsplit('_', 1)[-1]
            op2 = t2.name.rsplit('_', 1)[-1]
            assert not (base1 == base2 and op1 != op2 and op1 in ('encoder', 'decoder') and op2 in ('encoder', 'decoder'))
            # Char set check
            assert t1.range_chars & t2.domain_chars

def test_generate_chains_with_alternating_and_balancing(mock_tools):
    enforcer = RuleEnforcer([AlternatingRule, BalancingRule, RedundantPairRule, CharacterSetRule])
    chainer = ToolChainer(max_tree_size=4, rule_enforcer=enforcer, verbose=True)
    chainer.tools = mock_tools
    chainer.generate_chains()
    for chain in chainer.valid_chains:
        # Alternating: encoders followed by decoders and vice versa
        for i in range(len(chain) - 1):
            t1 = chainer.tools[chain[i]]
            t2 = chainer.tools[chain[i+1]]
            if '_encoder' in t1.name:
                assert '_decoder' in t2.name
            elif '_decoder' in t1.name:
                assert '_encoder' in t2.name
        # Balancing: even chains have equal enc/dec, odd chains differ by at most 1
        encode_count = sum(1 for i in chain if '_encoder' in chainer.tools[i].name)
        decode_count = sum(1 for i in chain if '_decoder' in chainer.tools[i].name)
        if len(chain) % 2 == 0:
            assert encode_count == decode_count
        else:
            assert abs(encode_count - decode_count) <= 1

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