import pytest
from pallas.tools.encoders.base64 import Base64Encoder
from pallas.common import EXTENDED_ASCII_CHARSET, BASE64_CHARSET

def test_base64_encoder_initialization():
    """Test Base64Encoder initialization."""
    encoder = Base64Encoder()
    assert encoder.name == "base64_encoder"
    assert "Converts ASCII text to Base64 encoding" in encoder.description

def test_base64_encoder_process():
    """Test Base64Encoder process method."""
    encoder = Base64Encoder()
    result = encoder._process("Hello")
    assert result == "SGVsbG8="

def test_base64_encoder_default_separator():
    encoder = Base64Encoder()
    result, sep, error = encoder.run("Hello")
    assert error is None
    assert result == "SGVsbG8="

def test_base64_encoder_empty_string():
    encoder = Base64Encoder()
    result, sep, error = encoder.run("")
    assert error is None
    assert result == ""

def test_base64_encoder_special_chars():
    encoder = Base64Encoder()
    result, sep, error = encoder.run("!@#$")
    assert error is None
    assert result == "IUAjJA=="

def test_base64_encoder_domain_chars():
    encoder = Base64Encoder()
    for i in range(256):
        assert chr(i) in encoder.domain_chars

def test_base64_encoder_range_chars():
    encoder = Base64Encoder()
    assert set(encoder.range_chars) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")