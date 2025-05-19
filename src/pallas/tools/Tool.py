from abc import ABC, abstractmethod
from typing import Set, Optional, Tuple
from pallas.tools.ToolError import ToolError

class Tool(ABC):
    """
    This is a generic tool class that can be used to create new tools.
    It provides a basic structure for tools, including name, description, domain characters, range characters, and separator.
    It also provides a run method that can be used to run the tool on an input string.
    It also provides a _process method that can be used to process the input string.
    The _process method is the core logic that concrete tools should implement.
    The run method is a wrapper around the _process method that provides error handling.
    The run method is the main method that should be used to run the tool.
    Always implement the _process method in the concrete tool class.
    Always implement the Tool interface when creating new tools.
    """
    def __init__(
        self,
        name: str,
        description: str,
        domain_chars: Set[str],
        range_chars: Set[str],
        separator: Optional[str] = None
    ):
        self.name = name
        self.description = description
        self.domain_chars = domain_chars
        self.range_chars = range_chars
        self.separator = separator

    def run(self, input_str: str, error: Optional[ToolError] = None) -> Tuple[str, Optional[ToolError]]:
        """
        Run the tool on the input string with error handling.

        Args:
            input_str: The input string to process
            error: Optional error from previous tool in chain

        Returns:
            Tuple of (result, error) where error is None on success
        """
        if error is not None:
            return (input_str, error)

        if not input_str:
            return ("", None)

        try:
            result = self._process(input_str)
            return (result, None)
        except Exception as e:
            return (input_str, ToolError(self.name, str(e)))

    @abstractmethod
    def _process(self, input_str: str) -> str:
        """
        Process the input string. This is the core logic that concrete tools should implement.

        Args:
            input_str: The input string to process

        Returns:
            The processed output string

        Raises:
            Exception: If there is an error during processing
        """
        pass