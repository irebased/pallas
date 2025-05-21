from pallas.tools.Tool import Tool
from pallas.common import EXTENDED_ASCII_CHARSET, HEX_CHARSET
from typing import Optional

class HexDecoder(Tool):
    name = "hex_decoder"
    description = "Converts hexadecimal representation back to ASCII text"
    domain_chars = HEX_CHARSET
    range_chars = EXTENDED_ASCII_CHARSET
    separator = " "  # Default separator is space

    def _process(self, input_str: str, input_separator: Optional[str] = None) -> tuple[str, Optional[str]]:
        """Convert hex representation back to ASCII text."""
        if not input_str:
            return ""
        result = []
        for hex_str in input_str.split(self.separator):
            # Convert to lowercase for consistency
            hex_str = hex_str.lower()
            value = int(hex_str, 16)
            if not 0 <= value <= 255:
                raise ValueError(f"Hex value {hex_str} is not a valid extended ASCII code (must be 00-FF)")
            result.append(chr(value))
        return "".join(result)