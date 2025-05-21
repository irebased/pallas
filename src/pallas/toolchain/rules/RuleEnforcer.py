from typing import List, Dict, Optional, Type
from pallas.toolchain.rules.ChainRule import ChainRule
from pallas.toolchain.ChainContext import ChainContext
from pallas.toolchain.rules.ChainRuleException import ChainRuleException

class RuleEnforcer:
    """Class responsible for enforcing chain rules and collecting statistics.

    This class:
    - Validates chains against a set of rules
    - Tracks statistics about rule violations
    - Provides methods to analyze pruning effectiveness
    """

    def __init__(self, rules: List[Type[ChainRule]]):
        """Initialize the rule enforcer.

        Args:
            rules: List of rule classes to enforce.
        """
        self.rules = rules
        # Initialize stats with rule class names as keys
        self.rule_stats: Dict[str, int] = {rule_class.__name__: 0 for rule_class in rules}
        self.total_validations = 0
        self.total_violations = 0

    def validate_chain_against_rules(self, chain_context: ChainContext) -> Optional[ChainRuleException]:
        """Validate a chain against all rules.

        Args:
            chain_context: The context containing information about the chain.

        Returns:
            ChainRuleException if any rule is violated, None otherwise.
        """
        self.total_validations += 1

        for rule_class in self.rules:
            if error := rule_class.validate(chain_context):
                self.rule_stats[rule_class.__name__] += 1
                self.total_violations += 1
                return error

        return None

    def get_rule_stats(self) -> Dict[str, int]:
        """Get statistics about rule violations.

        Returns:
            Dict mapping rule names to number of violations.
        """
        return self.rule_stats.copy()

    def get_violation_rate(self) -> float:
        """Get the rate of rule violations.

        Returns:
            Float between 0 and 1 representing the proportion of validations that resulted in violations.
        """
        if self.total_validations == 0:
            return 0.0
        return self.total_violations / self.total_validations

    def get_pruning_effectiveness(self, max_possible_nodes: int, actual_nodes: int) -> Dict[str, float]:
        """Calculate pruning effectiveness metrics.

        Args:
            max_possible_nodes: Maximum number of nodes that could have been visited.
            actual_nodes: Actual number of nodes visited.

        Returns:
            Dict containing:
            - pruning_rate: Proportion of nodes that were pruned
            - rule_contribution: Dict mapping rule names to their contribution to pruning
        """
        if max_possible_nodes == 0:
            return {
                'pruning_rate': 0.0,
                'rule_contribution': {rule_class.__name__: 0.0 for rule_class in self.rules}
            }

        # Calculate how many nodes we would have visited without pruning
        # This is based on the violation rate - if 60% of nodes violate rules,
        # then we would have visited 60% more nodes without pruning
        expected_nodes_without_pruning = int(actual_nodes * (1 + self.get_violation_rate()))

        # Calculate pruning rate based on how many nodes we avoided visiting
        pruning_rate = 1 - (actual_nodes / expected_nodes_without_pruning)

        # Calculate each rule's contribution to pruning
        total_violations = sum(self.rule_stats.values())
        if total_violations == 0:
            rule_contribution = {rule_class.__name__: 0.0 for rule_class in self.rules}
        else:
            rule_contribution = {
                rule_name: (violations / total_violations) * pruning_rate
                for rule_name, violations in self.rule_stats.items()
            }

        return {
            'pruning_rate': pruning_rate,
            'rule_contribution': rule_contribution
        }

    def format_stats(self, max_possible_nodes: int, actual_nodes: int) -> str:
        """Format statistics into a human-readable string.

        Args:
            max_possible_nodes: Maximum number of nodes that could have been visited.
            actual_nodes: Actual number of nodes visited.

        Returns:
            Formatted string containing all statistics.
        """
        effectiveness = self.get_pruning_effectiveness(max_possible_nodes, actual_nodes)

        stats = [
            "Rule Enforcement Statistics:",
            f"Total validations: {self.total_validations}",
            f"Total violations: {self.total_violations}",
            f"Violation rate: {self.get_violation_rate():.2%}",
            f"Pruning rate: {effectiveness['pruning_rate']:.2%}",
            "\nRule violations:"
        ]

        for rule_name, violations in self.rule_stats.items():
            contribution = effectiveness['rule_contribution'][rule_name]
            stats.append(f"  {rule_name}: {violations} violations ({contribution:.2%} of pruning)")

        return "\n".join(stats)