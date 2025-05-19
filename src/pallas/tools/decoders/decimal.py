from pallas.tools.Tool import Tool
from pallas.common import EXTENDED_ASCII_CHARSET, DECIMAL_CHARSET

class DecimalDecoder(Tool):
    def __init__(self, separator: str = " "):
        super().__init__(
            name="decimal_decoder",
            description="Converts decimal representation back to ASCII text",
            domain_chars=DECIMAL_CHARSET,
            range_chars=EXTENDED_ASCII_CHARSET,
            separator=separator
        )

    def _process(self, input_str: str) -> str:
        result = []
        for num in input_str.split(self.separator):
            value = int(num)
            if not 0 <= value <= 255:
                raise ValueError(f"Decimal value {value} is not a valid extended ASCII code (must be 0-255)")
            result.append(chr(value))
        return "".join(result)