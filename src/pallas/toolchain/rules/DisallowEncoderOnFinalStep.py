from pallas.toolchain.rules.ChainRule import ChainRule
from pallas.toolchain.ChainContext import ChainContext
from pallas.toolchain.rules.ChainRuleException import ChainRuleException

class DisallowEncoderOnFinalStepRule(ChainRule):
    """Rule that rejects chains that have an encoder on the final step.

    This rule ensures that:
    - The final step of the chain is not an encoder
    """

    @staticmethod
    def validate(chain_context: ChainContext) -> ChainRuleException | None:
        """Validate that the final step of the chain is not an encoder.

        Args:
            chain_context: The context containing information about the chain.

        Returns:
            ChainRuleException if the chain has an encoder on the final step, None otherwise.
        """
        if not chain_context.current_chain:
            return None

        if chain_context.target_length != len(chain_context.current_chain):
            return None

        if chain_context.next_tool is None:
            return None

        last_tool = chain_context.tools[chain_context.next_tool].name

        if last_tool.endswith('_encoder'):
            return ChainRuleException(chain_context=chain_context, message=f"Encoder on final step: {last_tool}")

        return None