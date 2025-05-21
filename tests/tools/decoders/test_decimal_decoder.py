import pytest
from pallas.tools.decoders.decimal import DecimalDecoder
from pallas.tools.ToolError import ToolError

def test_decimal_decoder_default_separator():
    decoder = DecimalDecoder()
    result, sep, error = decoder.run("72 101 108 108 111")
    assert error is None
    assert result == "Hello"

def test_decimal_decoder_custom_separator():
    decoder = DecimalDecoder(separator=",")
    result, sep, error = decoder.run("72,101,108,108,111")
    assert error is None
    assert result == "Hello"

def test_decimal_decoder_empty_string():
    decoder = DecimalDecoder()
    result, sep, error = decoder.run("")
    assert error is None
    assert result == ""

def test_decimal_decoder_special_chars():
    decoder = DecimalDecoder()
    result, sep, error = decoder.run("33 64 35 36")
    assert error is None
    assert result == "!@#$"

def test_decimal_decoder_invalid_input():
    decoder = DecimalDecoder()
    result, sep, error = decoder.run("72 101 108 108 111 999")
    assert error is not None
    assert isinstance(error, ToolError)
    assert "Decimal value 999 is not a valid extended ASCII code" in str(error)

def test_decimal_decoder_domain_chars():
    decoder = DecimalDecoder()
    assert set(decoder.domain_chars) == set("0123456789")

def test_decimal_decoder_range_chars():
    decoder = DecimalDecoder()
    for i in range(256):
        assert chr(i) in decoder.range_chars