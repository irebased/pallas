from pallas.tools.Tool import Tool
from pallas.common import EXTENDED_ASCII_CHARSET
from typing import Optional

class Reverse(Tool):
    name = "reverse"
    description = "Reverses the input string"
    domain_chars = EXTENDED_ASCII_CHARSET
    range_chars = EXTENDED_ASCII_CHARSET
    separator = None

    def _process(self, input_str: str, input_separator: Optional[str] = None) -> tuple[str, Optional[str]]:
        return input_str[::-1]