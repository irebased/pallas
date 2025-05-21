from pallas.tools.Tool import Tool
from pallas.common import EXTENDED_ASCII_CHARSET, DECIMAL_CHARSET
from typing import Optional

class DecimalEncoder(Tool):
    """Converts ASCII text to decimal representation."""

    name = "decimal_encoder"
    description = "Converts ASCII text to decimal representation"
    domain_chars = EXTENDED_ASCII_CHARSET
    range_chars = DECIMAL_CHARSET
    separator = " "  # Default separator

    def _process(self, input_str: str, input_separator: Optional[str] = None) -> tuple[str, Optional[str]]:
        """Convert ASCII text to decimal representation."""
        if not input_str:
            return ""
        return self.separator.join(str(ord(c)) for c in input_str)