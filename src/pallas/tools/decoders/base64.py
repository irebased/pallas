import base64
from pallas.tools.Tool import Tool
from pallas.common import EXTENDED_ASCII_CHARSET, BASE64_CHARSET

class Base64Decoder(Tool):
    def __init__(self):
        super().__init__(
            name="base64_decoder",
            description="Converts Base64 encoding back to ASCII text",
            domain_chars=BASE64_CHARSET,
            range_chars=EXTENDED_ASCII_CHARSET
        )

    def _process(self, input_str: str) -> str:
        # First decode the base64 string to bytes
        decoded_bytes = base64.b64decode(input_str)
        # Then interpret those bytes as latin1 characters
        return decoded_bytes.decode('latin1')