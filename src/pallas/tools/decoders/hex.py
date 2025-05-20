from pallas.tools.Tool import Tool
from pallas.common import EXTENDED_ASCII_CHARSET, HEX_CHARSET

class HexDecoder(Tool):
    def __init__(self, separator: str = " "):
        super().__init__(
            name="hex_decoder",
            description="Converts hexadecimal representation back to ASCII text",
            domain_chars=HEX_CHARSET,
            range_chars=EXTENDED_ASCII_CHARSET,
            separator=separator
        )

    def _process(self, input_str: str) -> str:
        if (input_str is None or input_str == ""):
            return ""
        result = []
        for hex_str in input_str.split(self.separator):
            value = int(hex_str, 16)
            if not 0 <= value <= 255:
                raise ValueError(f"Hex value {hex_str} is not a valid extended ASCII code (must be 00-FF)")
            result.append(chr(value))
        return "".join(result)