from pallas.toolchain.rules.ChainRule import ChainRule
from pallas.toolchain.ChainContext import ChainContext
from pallas.toolchain.rules.ChainRuleException import ChainRuleException

class RedundantPairRule(ChainRule):
    """Rule that prevents redundant encode-decode operations in a chain.

    This rule ensures that:
    - No adjacent tools in the chain are complementary (e.g., base64_encoder -> base64_decoder)
    - This prevents chains that would effectively cancel out their own operations
    """

    @staticmethod
    def validate(chain_context: ChainContext) -> ChainRuleException | None:
        """Validate that the chain does not contain redundant encode-decode pairs anywhere.

        Args:
            chain_context: The context containing information about the chain.

        Returns:
            ChainRuleException if the chain contains a redundant pair, None otherwise.
        """

        if not chain_context.current_chain or chain_context.current_chain == []:
            return None

        invalid_pairs = {
            'base64_encoder': 'base64_decoder',
            'base64_decoder': 'base64_encoder',
            'hex_encoder': 'hex_decoder',
            'hex_decoder': 'hex_encoder',
            'decimal_encoder': 'decimal_decoder',
            'decimal_decoder': 'decimal_encoder',
            'reverse': 'reverse',
        }

        last_chain_tool = chain_context.tools[chain_context.current_chain[-1]].name
        next_chain_tool = chain_context.tools[chain_context.next_tool].name

        if not last_chain_tool in invalid_pairs or not next_chain_tool in invalid_pairs:
            return None

        if invalid_pairs[last_chain_tool] == next_chain_tool:
            return ChainRuleException(chain_context=chain_context, message=f"Redundant pair: {last_chain_tool} -> {next_chain_tool}. \
Operation {last_chain_tool} followed by {next_chain_tool} would cancel out")

        return None