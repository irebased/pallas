import pytest
from pallas.tools.encoders.hex import HexEncoder
from pallas.common import EXTENDED_ASCII_CHARSET, HEX_CHARSET

def test_hex_encoder_initialization():
    """Test HexEncoder initialization."""
    encoder = HexEncoder()
    assert encoder.name == "hex_encoder"
    assert "Converts ASCII text to hexadecimal representation" in encoder.description
    assert encoder.domain_chars == EXTENDED_ASCII_CHARSET
    assert encoder.range_chars == HEX_CHARSET

def test_hex_encoder_process():
    """Test HexEncoder encoding functionality."""
    encoder = HexEncoder()

    # Test basic ASCII string
    assert encoder._process("Hello") == "48 65 6c 6c 6f"

    # Test empty string
    assert encoder._process("") == ""

    # Test string with special characters
    assert encoder._process("Hello, World!") == "48 65 6c 6c 6f 2c 20 57 6f 72 6c 64 21"

    # Test string with numbers
    assert encoder._process("123") == "31 32 33"

    # Test string with mixed content
    assert encoder._process("Hello123!") == "48 65 6c 6c 6f 31 32 33 21"

def test_hex_encoder_custom_separator():
    """Test HexEncoder with custom separator."""
    encoder = HexEncoder(separator=":")

    # Test with custom separator
    assert encoder._process("Hello") == "48:65:6c:6c:6f"

    # Test empty string
    assert encoder._process("") == ""