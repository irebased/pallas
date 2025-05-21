import pytest
from pallas.toolchain.rules.AlternatingRule import AlternatingRule
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
    assert AlternatingRule.validate(context) is None

def test_encoder_followed_by_encoder_is_invalid(mock_tools):
    """Test that an encoder followed by another encoder is invalid."""
    context = ChainContext(
        current_chain=['base64_encoder'],
        next_tool='hex_encoder',
        target_length=3,
        tools=mock_tools
    )
    error = AlternatingRule.validate(context)
    assert isinstance(error, ChainRuleException)
    assert "Encoder must be followed by decoder" in error.message

def test_decoder_followed_by_decoder_is_invalid(mock_tools):
    """Test that a decoder followed by another decoder is invalid."""
    context = ChainContext(
        current_chain=['base64_decoder'],
        next_tool='hex_decoder',
        target_length=3,
        tools=mock_tools
    )
    error = AlternatingRule.validate(context)
    assert isinstance(error, ChainRuleException)
    assert "Decoder must be followed by encoder" in error.message

def test_encoder_followed_by_decoder_is_valid(mock_tools):
    """Test that an encoder followed by a decoder is valid."""
    context = ChainContext(
        current_chain=['base64_encoder'],
        next_tool='hex_decoder',
        target_length=3,
        tools=mock_tools
    )
    assert AlternatingRule.validate(context) is None

def test_decoder_followed_by_encoder_is_valid(mock_tools):
    """Test that a decoder followed by an encoder is valid."""
    context = ChainContext(
        current_chain=['base64_decoder'],
        next_tool='hex_encoder',
        target_length=3,
        tools=mock_tools
    )
    assert AlternatingRule.validate(context) is None