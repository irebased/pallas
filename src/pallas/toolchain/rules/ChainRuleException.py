from pallas.toolchain.ChainContext import ChainContext

class ChainRuleException(ValueError):
    """Exception raised when a chain rule is violated."""

    def __init__(self, chain_context: ChainContext, message: str):
        self.chain_context = chain_context
        self.message = message
        super().__init__(self.message)