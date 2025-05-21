import pytest
from pallas.tools.decoders.hex import HexDecoder
from pallas.common import EXTENDED_ASCII_CHARSET, HEX_CHARSET

def test_hex_decoder_initialization():
    """Test HexDecoder initialization."""
    decoder = HexDecoder()
    assert decoder.name == "hex_decoder"
    assert "Converts hexadecimal representation back to ASCII text" in decoder.description

def test_hex_decoder_process():
    """Test HexDecoder process method."""
    decoder = HexDecoder()
    result = decoder._process("48 65 6c 6c 6f")
    assert result == "Hello"

def test_hex_decoder_custom_separator():
    """Test HexDecoder with custom separator."""
    decoder = HexDecoder(separator=",")
    result = decoder._process("48,65,6c,6c,6f")
    assert result == "Hello"

def test_hex_decoder_case_insensitive():
    """Test HexDecoder case insensitivity."""
    decoder = HexDecoder()
    result = decoder._process("48 65 6C 6C 6F")
    assert result == "Hello"

def test_hex_decoder_default_separator():
    decoder = HexDecoder()
    result, sep, error = decoder.run("48 65 6c 6c 6f")
    assert error is None
    assert result == "Hello"

def test_hex_decoder_empty_string():
    decoder = HexDecoder()
    result, sep, error = decoder.run("")
    assert error is None
    assert result == ""

def test_hex_decoder_special_chars():
    decoder = HexDecoder()
    result, sep, error = decoder.run("21 40 23 24")
    assert error is None
    assert result == "!@#$"

def test_hex_decoder_domain_chars():
    decoder = HexDecoder()
    assert set(decoder.domain_chars) == set("0123456789abcdef")

def test_hex_decoder_range_chars():
    decoder = HexDecoder()
    for i in range(256):
        assert chr(i) in decoder.range_chars