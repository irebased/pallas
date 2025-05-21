import pytest
from pallas.tools.encoders.octal import OctalEncoder

def test_octal_encoder_default_separator():
    encoder = OctalEncoder()
    result, sep, error = encoder.run("Hello")
    assert error is None
    assert result == "110 145 154 154 157"

def test_octal_encoder_custom_separator():
    encoder = OctalEncoder(separator=",")
    result, sep, error = encoder.run("Hello")
    assert error is None
    assert result == "110,145,154,154,157"

def test_octal_encoder_empty_string():
    encoder = OctalEncoder()
    result, sep, error = encoder.run("")
    assert error is None
    assert result == ""

def test_octal_encoder_special_chars():
    encoder = OctalEncoder()
    result, sep, error = encoder.run("!@#$")
    assert error is None
    assert result == "41 100 43 44"

def test_octal_encoder_domain_chars():
    encoder = OctalEncoder()
    for i in range(256):
        assert chr(i) in encoder.domain_chars

def test_octal_encoder_range_chars():
    encoder = OctalEncoder()
    assert set(encoder.range_chars) == set("01234567")