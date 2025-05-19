from pallas.tools.Tool import Tool
from pallas.common import EXTENDED_ASCII_CHARSET, DECIMAL_CHARSET

class DecimalEncoder(Tool):
    def __init__(self, separator: str = " "):
        super().__init__(
            name="decimal_encoder",
            description="Converts ASCII text to decimal representation",
            domain_chars=EXTENDED_ASCII_CHARSET,
            range_chars=DECIMAL_CHARSET,
            separator=separator
        )

    def _process(self, input_str: str) -> str:
        return self.separator.join(str(ord(c)) for c in input_str)