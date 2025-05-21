import pytest
from pallas.toolchain.rules.CharacterSetRule import CharacterSetRule
from pallas.toolchain.ChainContext import ChainContext
from pallas.toolchain.rules.ChainRuleException import ChainRuleException
from pallas.tools.Tool import Tool

class MockTool(Tool):
    """Mock tool for testing."""
    def __init__(self, name: str, domain_chars: set[str], range_chars: set[str]):
        self._name = name
        self._domain_chars = domain_chars
        self._range_chars = range_chars

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
        'tool1': MockTool('tool1', set('abc'), set('def')),
        'tool2': MockTool('tool2', set('def'), set('ghi')),
        'tool3': MockTool('tool3', set('ghi'), set('jkl')),
        'tool4': MockTool('tool4', set('xyz'), set('abc')),  # Incompatible with others
    }

def test_empty_chain_is_valid(mock_tools):
    """Test that an empty chain is always valid."""
    context = ChainContext(
        current_chain=[],
        next_tool='tool1',
        target_length=3,
        tools=mock_tools
    )
    assert CharacterSetRule.validate(context) is None

def test_compatible_chain_is_valid(mock_tools):
    """Test that a chain with compatible character sets is valid."""
    context = ChainContext(
        current_chain=['tool1', 'tool2'],
        next_tool='tool3',
        target_length=3,
        tools=mock_tools
    )
    assert CharacterSetRule.validate(context) is None

def test_incompatible_chain_is_invalid(mock_tools):
    """Test that a chain with incompatible character sets is invalid."""
    context = ChainContext(
        current_chain=['tool1', 'tool2'],
        next_tool='tool4',
        target_length=3,
        tools=mock_tools
    )
    error = CharacterSetRule.validate(context)
    assert isinstance(error, ChainRuleException)
    assert "incompatible with domain chars" in error.message

def test_single_tool_chain_is_valid(mock_tools):
    """Test that a chain with a single tool is valid."""
    context = ChainContext(
        current_chain=[],
        next_tool='tool1',
        target_length=1,
        tools=mock_tools
    )
    assert CharacterSetRule.validate(context) is None

def test_chain_with_matching_sets_is_valid(mock_tools):
    """Test that a chain where range matches domain is valid."""
    context = ChainContext(
        current_chain=['tool1'],
        next_tool='tool2',
        target_length=2,
        tools=mock_tools
    )
    assert CharacterSetRule.validate(context) is None