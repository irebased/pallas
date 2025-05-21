import pytest
from pallas.tools.encoders.hex import HexEncoder
from pallas.common import EXTENDED_ASCII_CHARSET, HEX_CHARSET

def test_hex_encoder_initialization():
    """Test HexEncoder initialization."""
    encoder = HexEncoder()
    assert encoder.name == "hex_encoder"
    assert "Converts ASCII text to hexadecimal representation" in encoder.description

def test_hex_encoder_process():
    """Test HexEncoder process method."""
    encoder = HexEncoder()
    result = encoder._process("Hello")
    assert result == "48 65 6c 6c 6f"

def test_hex_encoder_custom_separator():
    """Test HexEncoder with custom separator."""
    encoder = HexEncoder(separator=",")
    result = encoder._process("Hello")
    assert result == "48,65,6c,6c,6f"

def test_hex_encoder_default_separator():
    encoder = HexEncoder()
    result, sep, error = encoder.run("Hello")
    assert error is None
    assert result == "48 65 6c 6c 6f"

def test_hex_encoder_empty_string():
    encoder = HexEncoder()
    result, sep, error = encoder.run("")
    assert error is None
    assert result == ""

def test_hex_encoder_special_chars():
    encoder = HexEncoder()
    result, sep, error = encoder.run("!@#$")
    assert error is None
    assert result == "21 40 23 24"

def test_hex_encoder_domain_chars():
    encoder = HexEncoder()
    for i in range(256):
        assert chr(i) in encoder.domain_chars

def test_hex_encoder_range_chars():
    encoder = HexEncoder()
    assert set(encoder.range_chars) == set("0123456789abcdef")  # Only lowercase hex chars