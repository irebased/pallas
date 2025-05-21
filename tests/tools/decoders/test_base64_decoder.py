import pytest
from pallas.tools.decoders.base64 import Base64Decoder

def test_base64_decoder_default_separator():
    decoder = Base64Decoder()
    result, sep, error = decoder.run("SGVsbG8=")
    assert error is None
    assert result == "Hello"

def test_base64_decoder_empty_string():
    decoder = Base64Decoder()
    result, sep, error = decoder.run("")
    assert error is None
    assert result == ""

def test_base64_decoder_special_chars():
    decoder = Base64Decoder()
    result, sep, error = decoder.run("IUAjJA==")
    assert error is None
    assert result == "!@#$"

def test_base64_decoder_domain_chars():
    decoder = Base64Decoder()
    assert set(decoder.domain_chars) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")

def test_base64_decoder_range_chars():
    decoder = Base64Decoder()
    for i in range(256):
        assert chr(i) in decoder.range_chars