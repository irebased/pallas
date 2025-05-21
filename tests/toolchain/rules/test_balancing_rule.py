import pytest
from pallas.toolchain.rules.BalancingRule import BalancingRule
from pallas.toolchain.ChainContext import ChainContext
from pallas.toolchain.rules.ChainRuleException import ChainRuleException
from pallas.tools.Tool import Tool

class MockTool(Tool):
    """Mock tool for testing."""
    def __init__(self, name: str):
        self._name = name
        self._domain_chars = set('abc')
        self._range_chars = set('def')

    @property
    def name(self) -> str:
        return self._name

    @property
    def domain_chars(self) -> set[str]:
        return self._domain_chars

    @property
    def range_chars(self) -> set[str]:
        return self._range_chars

    def _process(self, input_text: str) -> str:
        return input_text

@pytest.fixture
def mock_tools():
    """Create mock tools for testing."""
    return {
        'base64_encoder': MockTool('base64_encoder'),
        'base64_decoder': MockTool('base64_decoder'),
        'hex_encoder': MockTool('hex_encoder'),
        'hex_decoder': MockTool('hex_decoder'),
    }

def test_empty_chain_is_valid(mock_tools):
    """Test that an empty chain is always valid."""
    context = ChainContext(
        current_chain=[],
        next_tool='base64_encoder',
        target_length=3,
        tools=mock_tools
    )
    assert BalancingRule.validate(context) is None

def test_even_length_chain_with_equal_counts_is_valid(mock_tools):
    """Test that an even-length chain with equal encoder/decoder counts is valid."""
    context = ChainContext(
        current_chain=['base64_encoder', 'hex_decoder'],
        next_tool='hex_encoder',
        target_length=3,
        tools=mock_tools
    )
    assert BalancingRule.validate(context) is None

def test_even_length_chain_with_unequal_counts_is_invalid(mock_tools):
    """Test that an even-length chain with unequal encoder/decoder counts is invalid."""
    context = ChainContext(
        current_chain=['base64_encoder', 'hex_encoder'],
        next_tool='base64_encoder',
        target_length=3,
        tools=mock_tools
    )
    error = BalancingRule.validate(context)
    assert isinstance(error, ChainRuleException)
    assert "Odd-length chains must have at most 1 more encoder than decoder or vice versa" in error.message

def test_odd_length_chain_with_diff_one_is_valid(mock_tools):
    """Test that an odd-length chain with difference of 1 is valid."""
    context = ChainContext(
        current_chain=['base64_encoder', 'hex_decoder'],
        next_tool='base64_encoder',
        target_length=3,
        tools=mock_tools
    )
    assert BalancingRule.validate(context) is None

def test_odd_length_chain_with_diff_two_is_invalid(mock_tools):
    """Test that an odd-length chain with difference of 2 is invalid."""
    context = ChainContext(
        current_chain=['base64_encoder', 'hex_encoder', 'base64_encoder'],
        next_tool='hex_encoder',
        target_length=4,
        tools=mock_tools
    )
    error = BalancingRule.validate(context)
    assert isinstance(error, ChainRuleException)
    assert "Even-length chains must have equal numbers of encoders and decoders" in error.message