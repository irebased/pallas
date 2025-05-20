import base64
from pallas.tools.Tool import Tool
from pallas.common import EXTENDED_ASCII_CHARSET, BASE64_CHARSET

class Base64Encoder(Tool):
    def __init__(self):
        super().__init__(
            name="base64_encoder",
            description="Converts ASCII text to Base64 encoding",
            domain_chars=EXTENDED_ASCII_CHARSET,
            range_chars=BASE64_CHARSET
        )

    def _process(self, input_str: str) -> str:
        return base64.b64encode(input_str.encode()).decode()