from pallas.tools.Tool import Tool
from pallas.common import EXTENDED_ASCII_CHARSET, OCTAL_CHARSET

class OctalEncoder(Tool):
    def __init__(self, separator: str = " "):
        super().__init__(
            name="octal_encoder",
            description="Converts ASCII text to octal representation",
            domain_chars=EXTENDED_ASCII_CHARSET,
            range_chars=OCTAL_CHARSET,
            separator=separator
        )

    def _process(self, input_str: str) -> str:
        return self.separator.join(f"{ord(c):o}" for c in input_str)