import pytest
from pallas.tools.encoders.base64 import Base64Encoder
from pallas.common import EXTENDED_ASCII_CHARSET, BASE64_CHARSET

def test_base64_encoder_initialization():
    """Test Base64Encoder initialization."""
    encoder = Base64Encoder()
    assert encoder.name == "base64_encoder"
    assert "Converts ASCII text to Base64 encoding" in encoder.description
    assert encoder.domain_chars == EXTENDED_ASCII_CHARSET
    assert encoder.range_chars == BASE64_CHARSET

def test_base64_encoder_process():
    """Test Base64Encoder encoding functionality."""
    encoder = Base64Encoder()

    # Test basic ASCII string
    assert encoder._process("Hello") == "SGVsbG8="

    # Test empty string
    assert encoder._process("") == ""

    # Test string with special characters
    assert encoder._process("Hello, World!") == "SGVsbG8sIFdvcmxkIQ=="

    # Test string with numbers
    assert encoder._process("123") == "MTIz"

    # Test string with mixed content
    assert encoder._process("Hello123!") == "SGVsbG8xMjMh"