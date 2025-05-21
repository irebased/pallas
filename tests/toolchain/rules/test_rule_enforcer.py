import pytest
from pallas.toolchain.rules.RuleEnforcer import RuleEnforcer
from pallas.toolchain.rules.ChainRule import ChainRule
from pallas.toolchain.ChainContext import ChainContext
from pallas.toolchain.rules.ChainRuleException import ChainRuleException

class MockRule(ChainRule):
    """A mock rule for testing."""
    __name__ = "MockRule"  # Add class name for RuleEnforcer

    @staticmethod
    def validate(chain_context: ChainContext) -> ChainRuleException | None:
        if getattr(chain_context, 'should_fail', False):
            return ChainRuleException(chain_context, f"Mock error from {getattr(chain_context, 'rule_name', 'MockRule')}")
        return None

@pytest.fixture
def mock_rules():
    """Create mock rules for testing."""
    return [MockRule, MockRule]  # Return class types instead of instances

@pytest.fixture
def chain_context():
    """Create a chain context for testing."""
    context = ChainContext(
        current_chain=[],
        next_tool=None,
        target_length=3,
        tools={}  # Empty tools dict since we're not using it in these tests
    )
    context.should_fail = False  # Add should_fail attribute
    context.rule_name = "MockRule"  # Add rule_name attribute
    return context

def test_rule_enforcer_initialization(mock_rules):
    """Test RuleEnforcer initialization."""
    enforcer = RuleEnforcer(mock_rules)
    assert enforcer.rules == mock_rules
    assert enforcer.rule_stats == {"MockRule": 0}
    assert enforcer.total_validations == 0
    assert enforcer.total_violations == 0

def test_validate_chain_against_rules_success(chain_context, mock_rules):
    """Test successful chain validation."""
    enforcer = RuleEnforcer(mock_rules)
    result = enforcer.validate_chain_against_rules(chain_context)
    assert result is None
    assert enforcer.total_validations == 1
    assert enforcer.total_violations == 0
    assert enforcer.rule_stats == {"MockRule": 0}

def test_validate_chain_against_rules_failure(chain_context, mock_rules):
    """Test failed chain validation."""
    chain_context.should_fail = True
    chain_context.rule_name = "MockRule"
    enforcer = RuleEnforcer(mock_rules)
    result = enforcer.validate_chain_against_rules(chain_context)
    assert isinstance(result, ChainRuleException)
    assert "Mock error from MockRule" in result.message
    assert enforcer.total_validations == 1
    assert enforcer.total_violations == 1
    assert enforcer.rule_stats == {"MockRule": 1}

def test_get_rule_stats(mock_rules):
    """Test getting rule statistics."""
    enforcer = RuleEnforcer(mock_rules)
    enforcer.rule_stats["MockRule"] = 5
    stats = enforcer.get_rule_stats()
    assert stats == {"MockRule": 5}
    # Verify that the returned dict is a copy
    stats["MockRule"] = 10
    assert enforcer.rule_stats["MockRule"] == 5

def test_get_violation_rate_no_validations(mock_rules):
    """Test violation rate with no validations."""
    enforcer = RuleEnforcer(mock_rules)
    assert enforcer.get_violation_rate() == 0.0

def test_get_violation_rate_with_validations(mock_rules):
    """Test violation rate with validations."""
    enforcer = RuleEnforcer(mock_rules)
    enforcer.total_validations = 10
    enforcer.total_violations = 3
    assert enforcer.get_violation_rate() == 0.3

def test_get_pruning_effectiveness_zero_nodes(mock_rules):
    """Test pruning effectiveness with zero nodes."""
    enforcer = RuleEnforcer(mock_rules)
    result = enforcer.get_pruning_effectiveness(0, 0)
    assert result["pruning_rate"] == 0.0
    assert result["rule_contribution"] == {"MockRule": 0.0}

def test_get_pruning_effectiveness_with_violations(mock_rules):
    """Test pruning effectiveness with violations."""
    enforcer = RuleEnforcer(mock_rules)
    enforcer.total_validations = 100
    enforcer.total_violations = 40
    enforcer.rule_stats["MockRule"] = 25
    result = enforcer.get_pruning_effectiveness(1000, 600)
    assert 0.0 <= result["pruning_rate"] <= 1.0
    assert sum(result["rule_contribution"].values()) == pytest.approx(result["pruning_rate"])

def test_format_stats(mock_rules):
    """Test formatting of statistics."""
    enforcer = RuleEnforcer(mock_rules)
    enforcer.total_validations = 100
    enforcer.total_violations = 40
    enforcer.rule_stats["MockRule"] = 25
    stats = enforcer.format_stats(1000, 600)
    assert "Rule Enforcement Statistics:" in stats
    assert "Total validations: 100" in stats
    assert "Total violations: 40" in stats
    assert "MockRule: 25 violations" in stats