import pytest
import base64
from pallas.tools.decoders.base64 import Base64Decoder
from pallas.common import EXTENDED_ASCII_CHARSET, BASE64_CHARSET

def test_base64_decoder_initialization():
    """Test Base64Decoder initialization."""
    decoder = Base64Decoder()
    assert decoder.name == "base64_decoder"
    assert "Converts Base64 encoding back to ASCII text" in decoder.description
    assert decoder.domain_chars == BASE64_CHARSET
    assert decoder.range_chars == EXTENDED_ASCII_CHARSET

def test_base64_decoder_process():
    """Test Base64Decoder decoding functionality."""
    decoder = Base64Decoder()

    # Test basic ASCII string
    assert decoder._process("SGVsbG8=") == "Hello"

    # Test empty string
    assert decoder._process("") == ""

    # Test string with special characters
    assert decoder._process("SGVsbG8sIFdvcmxkIQ==") == "Hello, World!"

    # Test string with numbers
    assert decoder._process("MTIz") == "123"

    # Test string with mixed content
    assert decoder._process("SGVsbG8xMjMh") == "Hello123!"

def test_base64_decoder_extended_ascii():
    """Test Base64Decoder with extended ASCII characters."""
    decoder = Base64Decoder()
    assert decoder._process("w6w=") == "Ã¬"