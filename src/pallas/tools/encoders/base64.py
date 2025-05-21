from pallas.tools.Tool import Tool
from pallas.common import EXTENDED_ASCII_CHARSET, BASE64_CHARSET
import base64
from typing import Optional

class Base64Encoder(Tool):
    """Converts ASCII text to Base64 encoding."""

    name = "base64_encoder"
    description = "Converts ASCII text to Base64 encoding"
    domain_chars = EXTENDED_ASCII_CHARSET
    range_chars = BASE64_CHARSET
    separator = None  # Base64 doesn't use separators

    def _process(self, input_str: str, input_separator: Optional[str] = None) -> tuple[str, Optional[str]]:
        """Convert ASCII text to Base64 encoding."""
        if not input_str:
            return ""
        return base64.b64encode(input_str.encode()).decode()