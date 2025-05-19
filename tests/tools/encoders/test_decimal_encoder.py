import pytest
from pallas.tools.encoders.decimal import DecimalEncoder

def test_decimal_encoder_default_separator():
    encoder = DecimalEncoder()
    result, error = encoder.run("Hello")
    assert error is None
    assert result == "72 101 108 108 111"

def test_decimal_encoder_custom_separator():
    encoder = DecimalEncoder(separator=",")
    result, error = encoder.run("Hello")
    assert error is None
    assert result == "72,101,108,108,111"

def test_decimal_encoder_empty_string():
    encoder = DecimalEncoder()
    result, error = encoder.run("")
    assert error is None
    assert result == ""

def test_decimal_encoder_special_chars():
    encoder = DecimalEncoder()
    result, error = encoder.run("!@#$")
    assert error is None
    assert result == "33 64 35 36"

def test_decimal_encoder_domain_chars():
    encoder = DecimalEncoder()
    for i in range(256):
        assert chr(i) in encoder.domain_chars

def test_decimal_encoder_range_chars():
    encoder = DecimalEncoder()
    assert encoder.range_chars == set("0123456789")