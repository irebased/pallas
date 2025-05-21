from pallas.tools.Tool import Tool
from pallas.common import EXTENDED_ASCII_CHARSET, DECIMAL_CHARSET
from typing import Optional

class DecimalDecoder(Tool):
    name = "decimal_decoder"
    description = "Converts decimal representation back to ASCII text"
    domain_chars = DECIMAL_CHARSET
    range_chars = EXTENDED_ASCII_CHARSET
    separator = " "  # Default separator is space

    def _process(self, input_str: str, input_separator: Optional[str] = None) -> tuple[str, Optional[str]]:
        """Convert decimal representation back to ASCII text."""
        if not input_str:
            return ""
        result = []
        for dec_str in input_str.split(self.separator):
            value = int(dec_str)
            if not 0 <= value <= 255:
                raise ValueError(f"Decimal value {dec_str} is not a valid extended ASCII code (must be 0-255)")
            result.append(chr(value))
        return "".join(result)