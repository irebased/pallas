from pallas.toolchain.rules.ChainRule import ChainRule
from pallas.toolchain.ChainContext import ChainContext
from pallas.toolchain.rules.ChainRuleException import ChainRuleException

class CharacterSetRule(ChainRule):
    """Rule that enforces character set compatibility between adjacent tools.

    This rule ensures that:
    - The range characters of the current tool are compatible with the domain characters
      of the next tool
    - Compatibility means either:
      - Range chars are a subset of domain chars
      - Range chars are a superset of domain chars
      - Range chars equal domain chars
    """

    @staticmethod
    def validate(chain_context: ChainContext) -> ChainRuleException | None:
        """Validate that adjacent tools have compatible character sets.

        Args:
            chain_context: The context containing information about the chain.

        Returns:
            ChainRuleException if the chain violates the character set rule, None otherwise.
        """
        if not chain_context.current_chain:
            return None

        current_tool = chain_context.tools[chain_context.current_chain[-1]]
        next_tool = chain_context.tools[chain_context.next_tool]

        range_chars = current_tool.range_chars
        domain_chars = next_tool.domain_chars

        is_valid = (range_chars.issubset(domain_chars) or
                   range_chars.issuperset(domain_chars) or
                   range_chars == domain_chars)

        if not is_valid:
            return ChainRuleException(
                f"Character set mismatch: {current_tool.name} -> {next_tool.name}",
                f"Range chars ({sorted(range_chars)}) incompatible with domain chars ({sorted(domain_chars)})"
            )

        return None