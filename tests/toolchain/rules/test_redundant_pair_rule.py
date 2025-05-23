import pytest
from pallas.toolchain.rules.RedundantPairRule import RedundantPairRule
from pallas.toolchain.ChainContext import ChainContext
from pallas.toolchain.rules.ChainRuleException import ChainRuleException
from pallas.tools.Tool import Tool

class MockTool(Tool):
    """Mock tool for testing."""
    def __init__(self, name, operation_type):
        super().__init__()
        self.name = name
        self.operation_type = operation_type
        self.description = f"Mock tool {name}"
        self.domain_chars = set('abc')
        self.range_chars = set('def')

    def _process(self, input_str: str) -> str:
        return input_str

@pytest.fixture
def mock_tools():
    """Create a set of mock tools for testing."""
    return {
        'base64_encoder': MockTool('base64_encoder', 'encode'),
        'base64_decoder': MockTool('base64_decoder', 'decode'),
        'hex_encoder': MockTool('hex_encoder', 'encode'),
        'hex_decoder': MockTool('hex_decoder', 'decode'),
        'other_tool': MockTool('other_tool', 'other')
    }

@pytest.fixture
def chain_context(mock_tools):
    """Create a chain context for testing."""
    return ChainContext(
        current_chain=[],
        next_tool=None,
        target_length=3,
        tools=mock_tools
    )

def test_redundant_pair_rule_no_pairs(chain_context, mock_tools):
    """Test that non-redundant chains are allowed."""
    chain_context.current_chain = ['hex_encoder']
    chain_context.next_tool = 'base64_encoder'
    chain_context.tools = mock_tools
    result = RedundantPairRule.validate(chain_context)
    assert result is None

def test_redundant_pair_rule_encoder_decoder(chain_context, mock_tools):
    """Test that encoder followed by decoder is caught."""
    chain_context.current_chain = ['base64_encoder']
    chain_context.next_tool = 'base64_decoder'
    chain_context.tools = mock_tools
    result = RedundantPairRule.validate(chain_context)
    assert result is not None
    assert "Operation base64_encoder followed by base64_decoder would cancel out" in result.message

def test_redundant_pair_rule_decoder_encoder(chain_context, mock_tools):
    """Test that decoder followed by encoder is caught."""
    chain_context.current_chain = ['base64_decoder']
    chain_context.next_tool = 'base64_encoder'
    chain_context.tools = mock_tools
    result = RedundantPairRule.validate(chain_context)
    assert result is not None
    assert "Operation base64_decoder followed by base64_encoder would cancel out" in result.message

def test_redundant_pair_rule_middle_of_chain(chain_context, mock_tools):
    """Test that redundant pairs are caught in the middle of a chain."""
    chain_context.current_chain = ['hex_encoder', 'base64_encoder']
    chain_context.next_tool = 'base64_decoder'
    chain_context.tools = mock_tools
    result = RedundantPairRule.validate(chain_context)
    assert result is not None
    assert "Operation base64_encoder followed by base64_decoder would cancel out" in result.message

def test_redundant_pair_rule_custom_tool(chain_context, mock_tools):
    """Test that custom tools without encoder/decoder suffix are allowed."""
    chain_context.current_chain = ['other_tool']
    chain_context.next_tool = 'base64_encoder'
    chain_context.tools = mock_tools
    result = RedundantPairRule.validate(chain_context)
    assert result is None

def test_redundant_pair_rule_same_operation(chain_context, mock_tools):
    """Test that same operations (encoder->encoder) are allowed."""
    chain_context.current_chain = ['base64_encoder']
    chain_context.next_tool = 'hex_encoder'
    chain_context.tools = mock_tools
    result = RedundantPairRule.validate(chain_context)
    assert result is None

def test_redundant_pair_rule_empty_chain(chain_context, mock_tools):
    """Test that empty chains are allowed."""
    chain_context.current_chain = []
    chain_context.next_tool = 'base64_encoder'
    chain_context.tools = mock_tools
    result = RedundantPairRule.validate(chain_context)
    assert result is None

def test_empty_chain_is_valid(mock_tools):
    """Test that an empty chain is always valid."""
    context = ChainContext(
        current_chain=[],
        next_tool='base64_encoder',
        target_length=3,
        tools=mock_tools
    )
    assert RedundantPairRule.validate(context) is None

def test_single_tool_chain_is_valid(mock_tools):
    """Test that a chain with a single tool is valid."""
    context = ChainContext(
        current_chain=['base64_encoder'],
        next_tool='hex_encoder',
        target_length=2,
        tools=mock_tools
    )
    assert RedundantPairRule.validate(context) is None

def test_redundant_pair_is_invalid(mock_tools):
    """Test that a chain with redundant encode-decode pairs is invalid."""
    context = ChainContext(
        current_chain=['base64_encoder'],
        next_tool='base64_decoder',
        target_length=2,
        tools=mock_tools
    )
    result = RedundantPairRule.validate(context)
    assert isinstance(result, ChainRuleException)
    assert "redundant pair" in result.message.lower()

def test_non_redundant_pair_is_valid(mock_tools):
    """Test that a chain with non-redundant encode-decode pairs is valid."""
    context = ChainContext(
        current_chain=['base64_encoder', 'hex_encoder'],
        next_tool='other_tool',
        target_length=3,
        tools=mock_tools
    )
    result = RedundantPairRule.validate(context)
    assert result is None

def test_same_operation_is_valid(mock_tools):
    """Test that a chain with same operations is valid."""
    context = ChainContext(
        current_chain=['base64_encoder'],
        next_tool='hex_encoder',
        target_length=2,
        tools=mock_tools
    )
    result = RedundantPairRule.validate(context)
    assert result is None

def test_other_operation_is_valid(mock_tools):
    """Test that a chain with other operations is valid."""
    context = ChainContext(
        current_chain=['other_tool'],
        next_tool='base64_encoder',
        target_length=2,
        tools=mock_tools
    )
    result = RedundantPairRule.validate(context)
    assert result is None