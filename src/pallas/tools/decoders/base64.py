from pallas.tools.Tool import Tool
from pallas.common import EXTENDED_ASCII_CHARSET, BASE64_CHARSET
import base64
from typing import Optional

class Base64Decoder(Tool):
    name = "base64_decoder"
    description = "Converts Base64 representation back to ASCII text"
    domain_chars = BASE64_CHARSET
    range_chars = EXTENDED_ASCII_CHARSET
    separator = None  # Base64 doesn't use separators

    def _process(self, input_str: str, input_separator: Optional[str] = None) -> tuple[str, Optional[str]]:
        """Convert Base64 representation back to ASCII text."""
        if not input_str:
            return ""
        try:
            if input_separator:
                input_str = input_str.strip().replace(input_separator, '')

            # if the base64 has invalid padding, add the necessary amount of padding before processing (===)
            if len(input_str) % 4 != 0:
                input_str += '=' * (4 - len(input_str) % 4)
            return base64.b64decode(input_str.encode()).decode('latin1'), None
        except Exception as e:
            raise ValueError(f"Invalid Base64 input: {str(e)}")