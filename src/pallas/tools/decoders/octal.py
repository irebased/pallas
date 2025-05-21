from pallas.tools.Tool import Tool
from pallas.common import EXTENDED_ASCII_CHARSET, OCTAL_CHARSET
from typing import Optional

class OctalDecoder(Tool):
    name = "octal_decoder"
    description = "Converts octal representation back to ASCII text"
    domain_chars = OCTAL_CHARSET
    range_chars = EXTENDED_ASCII_CHARSET
    separator = " "  # Default separator is space

    def _process(self, input_str: str, input_separator: Optional[str] = None) -> tuple[str, Optional[str]]:
        """Convert octal representation back to ASCII text."""
        if not input_str:
            return ""
        result = []
        for oct_str in input_str.split(self.separator):
            value = int(oct_str, 8)
            if not 0 <= value <= 255:
                raise ValueError(f"Octal value {oct_str} is not a valid extended ASCII code (must be 000-377)")
            result.append(chr(value))
        return "".join(result)