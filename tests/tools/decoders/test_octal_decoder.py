import pytest
from pallas.tools.decoders.octal import OctalDecoder
from pallas.tools.ToolError import ToolError

def test_octal_decoder_default_separator():
    decoder = OctalDecoder()
    result, sep, error = decoder.run("110 145 154 154 157")
    assert error is None
    assert result == "Hello"

def test_octal_decoder_custom_separator():
    decoder = OctalDecoder(separator=",")
    result, sep, error = decoder.run("110,145,154,154,157")
    assert error is None
    assert result == "Hello"

def test_octal_decoder_empty_string():
    decoder = OctalDecoder()
    result, sep, error = decoder.run("")
    assert error is None
    assert result == ""

def test_octal_decoder_special_chars():
    decoder = OctalDecoder()
    result, sep, error = decoder.run("41 100 43 44")
    assert error is None
    assert result == "!@#$"

def test_octal_decoder_invalid_input():
    decoder = OctalDecoder()
    result, sep, error = decoder.run("110 145 154 154 157 999")
    assert error is not None
    assert isinstance(error, ToolError)
    assert "Input contains invalid characters" in str(error)

def test_octal_decoder_out_of_range():
    decoder = OctalDecoder()
    result, sep, error = decoder.run("400")  # 400 octal = 256 decimal, which is out of range
    assert error is not None
    assert isinstance(error, ToolError)
    assert "Octal value 400 is not a valid extended ASCII code" in str(error)

def test_octal_decoder_domain_chars():
    decoder = OctalDecoder()
    assert set(decoder.domain_chars) == set("01234567")

def test_octal_decoder_range_chars():
    decoder = OctalDecoder()
    for i in range(256):
        assert chr(i) in decoder.range_chars