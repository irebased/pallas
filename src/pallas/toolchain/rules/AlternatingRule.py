from pallas.toolchain.rules.ChainRule import ChainRule
from pallas.toolchain.ChainContext import ChainContext
from pallas.toolchain.rules.ChainRuleException import ChainRuleException

class AlternatingRule(ChainRule):
    """Rule that enforces alternating encoder/decoder operations in a chain.

    This rule ensures that:
    - If the current tool is an encoder, the next tool must be a decoder
    - If the current tool is a decoder, the next tool must be an encoder
    """

    @staticmethod
    def validate(chain_context: ChainContext) -> ChainRuleException | None:
        """Validate that tools alternate between encoder and decoder.

        Args:
            chain_context: The context containing information about the chain.

        Returns:
            ChainRuleException if the chain violates the alternating rule, None otherwise.
        """
        if not chain_context.current_chain:
            return None

        current_tool = chain_context.tools[chain_context.current_chain[-1]]
        next_tool = chain_context.tools[chain_context.next_tool]

        # Get operation types
        current_is_encoder = '_encoder' in current_tool.name
        current_is_decoder = '_decoder' in current_tool.name
        next_is_encoder = '_encoder' in next_tool.name
        next_is_decoder = '_decoder' in next_tool.name

        # If current tool is an encoder, next must be a decoder
        if current_is_encoder and next_is_encoder:
            return ChainRuleException(chain_context=chain_context, message=f"Non-alternating chain: {current_tool.name} -> {next_tool.name}. \
Encoder must be followed by decoder")

        # If current tool is a decoder, next must be an encoder
        if current_is_decoder and next_is_decoder:
            return ChainRuleException(chain_context=chain_context, message=f"Non-alternating chain: {current_tool.name} -> {next_tool.name}. \
Decoder must be followed by encoder")

        return None