import pytest
from pallas.tools.decoders.hex import HexDecoder
from pallas.common import EXTENDED_ASCII_CHARSET, HEX_CHARSET

def test_hex_decoder_initialization():
    """Test HexDecoder initialization."""
    decoder = HexDecoder()
    assert decoder.name == "hex_decoder"
    assert "Converts hexadecimal representation back to ASCII text" in decoder.description
    assert decoder.domain_chars == HEX_CHARSET
    assert decoder.range_chars == EXTENDED_ASCII_CHARSET

def test_hex_decoder_process():
    """Test HexDecoder decoding functionality."""
    decoder = HexDecoder()

    # Test basic ASCII string
    assert decoder._process("48 65 6c 6c 6f") == "Hello"

    # Test empty string
    assert decoder._process("") == ""

    # Test string with special characters
    assert decoder._process("48 65 6c 6c 6f 2c 20 57 6f 72 6c 64 21") == "Hello, World!"

    # Test string with numbers
    assert decoder._process("31 32 33") == "123"

    # Test string with mixed content
    assert decoder._process("48 65 6c 6c 6f 31 32 33 21") == "Hello123!"

def test_hex_decoder_custom_separator():
    """Test HexDecoder with custom separator."""
    decoder = HexDecoder(separator=":")

    # Test with custom separator
    assert decoder._process("48:65:6c:6c:6f") == "Hello"

    # Test empty string
    assert decoder._process("") == ""

def test_hex_decoder_case_insensitive():
    """Test HexDecoder with mixed case hex values."""
    decoder = HexDecoder()

    # Test mixed case hex values
    assert decoder._process("48 65 6C 6c 6F") == "Hello"
    assert decoder._process("48 65 6c 6C 6f") == "Hello"