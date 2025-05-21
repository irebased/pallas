from abc import ABC, abstractmethod
from pallas.toolchain.ChainContext import ChainContext
from pallas.toolchain.rules.ChainRuleException import ChainRuleException
from typing import Optional

class ChainRule(ABC):
    """Abstract base class for chain rules.

    Any class implementing ChainRule must implement the validate method.
    The validate method should return None if the chain is valid according to the rule,
    and a ChainRuleException otherwise.
    """

    @staticmethod
    @abstractmethod
    def validate(chain_context: ChainContext) -> Optional[ChainRuleException]:
        """Validate the chain according to this rule.

        Args:
            chain_context: The context containing information about the chain to validate.

        Returns:
            Optional[ChainRuleException]: An exception if the chain is invalid, None otherwise.
        """
        return