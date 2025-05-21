from pallas.tools.Tool import Tool
from pallas.common import EXTENDED_ASCII_CHARSET, HEX_CHARSET
from typing import Optional

class HexEncoder(Tool):
    name = "hex_encoder"
    description = "Converts ASCII text to hexadecimal representation"
    domain_chars = EXTENDED_ASCII_CHARSET
    range_chars = HEX_CHARSET
    separator = " "  # Default separator is space

    def _process(self, input_str: str, input_separator: Optional[str] = None) -> tuple[str, Optional[str]]:
        """Convert input string to hex representation."""
        if not input_str:
            return ""
        return self.separator.join(hex(ord(c))[2:].zfill(2) for c in input_str)