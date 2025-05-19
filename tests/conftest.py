import pytest
from pallas.tools.encoders.octal import OctalEncoder
from pallas.tools.encoders.decimal import DecimalEncoder
from pallas.tools.decoders.octal import OctalDecoder
from pallas.tools.decoders.decimal import DecimalDecoder

@pytest.fixture
def octal_encoder():
    return OctalEncoder()

@pytest.fixture
def decimal_encoder():
    return DecimalEncoder()

@pytest.fixture
def octal_decoder():
    return OctalDecoder()

@pytest.fixture
def decimal_decoder():
    return DecimalDecoder()