from pallas.tools.Tool import Tool
from pallas.common import EXTENDED_ASCII_CHARSET, HEX_CHARSET

class HexEncoder(Tool):
    def __init__(self, separator: str = " "):
        super().__init__(
            name="hex_encoder",
            description="Converts ASCII text to hexadecimal representation",
            domain_chars=EXTENDED_ASCII_CHARSET,
            range_chars=HEX_CHARSET,
            separator=separator
        )

    def _process(self, input_str: str) -> str:
        return self.separator.join(hex(ord(c))[2:].zfill(2) for c in input_str)