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
        # Helper functions
        def get_base_name(name: str) -> str:
            if name.endswith('_encoder'):
                return name[:-8]
            if name.endswith('_decoder'):
                return name[:-8]
            return name

        def get_operation(name: str) -> str:
            if name.endswith('_encoder'):
                return 'encode'
            if name.endswith('_decoder'):
                return 'decode'
            return 'unknown'

        # Build the full chain including the next tool
        full_chain = chain_context.current_chain + [chain_context.next_tool]
        tools = chain_context.tools

        # Check all adjacent pairs
        for i in range(len(full_chain) - 1):
            t1 = tools[full_chain[i]]
            t2 = tools[full_chain[i+1]]
            base1 = get_base_name(t1.name)
            base2 = get_base_name(t2.name)
            op1 = get_operation(t1.name)
            op2 = get_operation(t2.name)
            if base1 == base2 and op1 != op2 and op1 != 'unknown' and op2 != 'unknown':
                return ChainRuleException(chain_context=chain_context, message=f"Redundant encode-decode pair: {t1.name} -> {t2.name}. \
Operation {op1} followed by {op2} would cancel out")
        return None