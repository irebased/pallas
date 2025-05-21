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
    name: str = "tool"
    description: str = "A tool"
    domain_chars: str = ""
    range_chars: str = ""
    separator: Optional[str] = None

    def __init__(self, separator: Optional[str] = None):
        """Initialize the tool with an optional custom separator."""
        if separator is not None:
            self.separator = separator

    @abstractmethod
    def _process(self, input_str: str, input_separator: Optional[str] = None) -> tuple[str, Optional[str]]:
        """
        Process the input string. This is the core logic that concrete tools should implement.
        This method should not perform any input validation - that is handled by run().

        Args:
            input_str: The input string to process
            input_separator: The separator to use for the input string

        Returns:
            The processed output string and the separator to use for the output string

        Raises:
            Exception: If there is an error during processing
        """
        pass

    def run(self, input_str: str, input_separator: Optional[str] = None, error: Optional[ToolError] = None) -> Tuple[str, Optional[str], Optional[ToolError]]:
        """Run the tool on the input string.

        Args:
            input_str: The input string to process
            input_separator: The separator to use for the input string
            error: Optional error from a previous tool

        Returns:
            A tuple of (result, error) where result is the processed string and error is None if successful
        """
        if error is not None:
            return input_str, self.separator, error

        # Validate input characters
        if input_str:
            invalid_chars = set(input_str) - set(self.domain_chars)
            if self.separator and self.separator in invalid_chars:
                invalid_chars.remove(self.separator)
            if input_separator and input_separator in invalid_chars:
                invalid_chars.remove(input_separator)
            if invalid_chars:
                return input_str, self.separator, ToolError(self.name, f"Input contains invalid characters: {invalid_chars}")

        try:
            result = self._process(input_str, input_separator)

            return result, self.separator, None
        except Exception as e:
            return input_str, self.separator, ToolError(self.name, str(e))