import pytest
from pallas.toolchain.rules.ChainRule import ChainRule
from pallas.toolchain.ChainContext import ChainContext
from pallas.toolchain.rules.ChainRuleException import ChainRuleException

class MockChainRule(ChainRule):
    """A concrete implementation of ChainRule for testing."""
    def __init__(self, should_fail: bool = False, error_message: str = "Test error"):
        self.should_fail = should_fail
        self.error_message = error_message

    @staticmethod
    def validate(chain_context: ChainContext) -> ChainRuleException | None:
        if chain_context.should_fail:
            return ChainRuleException(chain_context, chain_context.error_message)
        return None

def test_chain_rule_abstract_base():
    """Test that ChainRule cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ChainRule()

def test_chain_rule_validate_abstract():
    """Test that validate is an abstract method."""
    # Create a subclass without implementing validate
    class IncompleteRule(ChainRule):
        pass

    with pytest.raises(TypeError):
        IncompleteRule()

def test_mock_chain_rule_success():
    """Test successful validation with mock rule."""
    context = ChainContext(
        current_chain=[],
        next_tool=None,
        target_length=3,
        tools={}
    )
    context.should_fail = False
    rule = MockChainRule()
    result = rule.validate(context)
    assert result is None

def test_mock_chain_rule_failure():
    """Test failed validation with mock rule."""
    context = ChainContext(
        current_chain=[],
        next_tool=None,
        target_length=3,
        tools={}
    )
    context.should_fail = True
    context.error_message = "Test error message"
    rule = MockChainRule()
    result = rule.validate(context)
    assert isinstance(result, ChainRuleException)
    assert result.message == "Test error message"