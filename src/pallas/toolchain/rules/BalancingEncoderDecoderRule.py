from pallas.toolchain.rules.ChainRule import ChainRule
from pallas.toolchain.ChainContext import ChainContext
from pallas.toolchain.rules.ChainRuleException import ChainRuleException

class BalancingEncoderDecoderRule(ChainRule):
    """Rule that enforces balanced encoder/decoder operations in a chain.

    This rule ensures that:
    - For even-length chains: equal number of encoders and decoders
    - For odd-length chains: difference between encoders and decoders is at most 1
    """

    @staticmethod
    def validate(chain_context: ChainContext) -> ChainRuleException | None:
        """Validate that the chain has balanced encoder/decoder operations.

        Args:
            chain_context: The context containing information about the chain.

        Returns:
            ChainRuleException if the chain violates the balancing rule, None otherwise.
        """
        if not chain_context.current_chain or chain_context.current_chain == []:
            return None

        encode_count = sum(1 for i in chain_context.current_chain
                         if '_encoder' in chain_context.tools[i].name)
        decode_count = sum(1 for i in chain_context.current_chain
                         if '_decoder' in chain_context.tools[i].name)
        other_count = encode_count + decode_count - len(chain_context.current_chain)

        if chain_context.next_tool is not None:
            if '_encoder' in chain_context.tools[chain_context.next_tool].name:
                encode_count += 1
            elif '_decoder' in chain_context.tools[chain_context.next_tool].name:
                decode_count += 1

        total_length = len(chain_context.current_chain) + (1 if chain_context.next_tool is not None else 0) - other_count
        diff = abs(encode_count - decode_count)

        if total_length % 2 == 0:
            if encode_count != decode_count:
                return ChainRuleException(chain_context=chain_context, message=f"Unbalanced chain: {encode_count} encoders, {decode_count} decoders. \
Even-length chains must have equal numbers of encoders and decoders for chain: {chain_context.print_chain()} and next tool: {chain_context.print_next_tool()}")
        else:
            if diff > 1:
                return ChainRuleException(chain_context=chain_context, message=f"Unbalanced chain: {encode_count} encoders, {decode_count} decoders. \
Odd-length chains must have at most 1 more encoder than decoder or vice versa for chain: {chain_context.print_chain()} and next tool: {chain_context.print_next_tool()}")

        return None