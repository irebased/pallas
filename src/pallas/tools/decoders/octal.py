from pallas.tools.Tool import Tool
from pallas.common import EXTENDED_ASCII_CHARSET, OCTAL_CHARSET

class OctalDecoder(Tool):
    def __init__(self, separator: str = " "):
        super().__init__(
            name="octal_decoder",
            description="Converts octal representation back to ASCII text",
            domain_chars=OCTAL_CHARSET,
            range_chars=EXTENDED_ASCII_CHARSET,
            separator=separator
        )

    def _process(self, input_str: str) -> str:
        result = []
        for part in input_str.split(self.separator):
            value = int(part, 8)
            if not 0 <= value <= 255:
                raise ValueError(f"Octal value {part} ({value} decimal) is not a valid extended ASCII code (must be 0-377 octal)")
            result.append(chr(value))
        return "".join(result)