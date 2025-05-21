from pallas.tools.Tool import Tool
from pallas.common import EXTENDED_ASCII_CHARSET, OCTAL_CHARSET
from typing import Optional

class OctalEncoder(Tool):
    """Converts ASCII text to octal representation."""

    name = "octal_encoder"
    description = "Converts ASCII text to octal representation"
    domain_chars = EXTENDED_ASCII_CHARSET
    range_chars = OCTAL_CHARSET
    separator = " "  # Default separator

    def _process(self, input_str: str, input_separator: Optional[str] = None) -> tuple[str, Optional[str]]:
        """Convert ASCII text to octal representation."""
        if not input_str:
            return ""
        return self.separator.join(f"{ord(c):o}" for c in input_str)