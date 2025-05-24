import pytest
from pallas.toolchain.rules.DisallowEncoderOnFinalStep import DisallowEncoderOnFinalStepRule
from pallas.toolchain.ChainContext import ChainContext
from pallas.toolchain.rules.ChainRuleException import ChainRuleException
from pallas.tools.encoders.base64 import Base64Encoder
from pallas.tools.decoders.base64 import Base64Decoder
from pallas.tools.transformers.reverse import Reverse

@pytest.fixture
def chain_context():
    """Create a basic chain context for testing."""
    return ChainContext(
        current_chain=[],
        next_tool="",
        target_length=3,
        tools={}
    )

def test_validate_empty_chain(chain_context):
    """Test that an empty chain is valid."""
    chain_context.current_chain = []
    chain_context.next_tool = "base64_encoder"
    chain_context.tools = {"base64_encoder": Base64Encoder()}

    result = DisallowEncoderOnFinalStepRule.validate(chain_context)
    assert result is None

def test_validate_non_encoder_final_step(chain_context):
    """Test that a chain ending with a non-encoder tool is valid."""
    chain_context.current_chain = ["base64_encoder"]
    chain_context.next_tool = "reverse"
    chain_context.tools = {
        "base64_encoder": Base64Encoder(),
        "reverse": Reverse()
    }

    result = DisallowEncoderOnFinalStepRule.validate(chain_context)
    assert result is None

def test_validate_encoder_final_step(chain_context):
    """Test that a chain ending with an encoder tool is invalid."""
    chain_context.current_chain = ["reverse"]
    chain_context.next_tool = "base64_encoder"
    chain_context.target_length = 1
    chain_context.tools = {
        "reverse": Reverse(),
        "base64_encoder": Base64Encoder()
    }

    result = DisallowEncoderOnFinalStepRule.validate(chain_context)
    assert isinstance(result, ChainRuleException)
    assert "Encoder on final step: base64_encoder" in result.message

def test_validate_decoder_final_step(chain_context):
    """Test that a chain ending with a decoder tool is valid."""
    chain_context.current_chain = ["base64_encoder"]
    chain_context.next_tool = "base64_decoder"
    chain_context.tools = {
        "base64_encoder": Base64Encoder(),
        "base64_decoder": Base64Decoder()
    }

    result = DisallowEncoderOnFinalStepRule.validate(chain_context)
    assert result is None

def test_validate_multiple_steps_with_encoder_final(chain_context):
    """Test that a longer chain ending with an encoder is invalid."""
    chain_context.current_chain = ["reverse", "base64_decoder"]
    chain_context.next_tool = "base64_encoder"
    chain_context.target_length = 2
    chain_context.tools = {
        "reverse": Reverse(),
        "base64_decoder": Base64Decoder(),
        "base64_encoder": Base64Encoder()
    }

    result = DisallowEncoderOnFinalStepRule.validate(chain_context)
    assert isinstance(result, ChainRuleException)
    assert "Encoder on final step: base64_encoder" in result.message

def test_validate_multiple_steps_without_encoder_final(chain_context):
    """Test that a longer chain not ending with an encoder is valid."""
    chain_context.current_chain = ["base64_encoder", "reverse"]
    chain_context.next_tool = "base64_decoder"
    chain_context.tools = {
        "base64_encoder": Base64Encoder(),
        "reverse": Reverse(),
        "base64_decoder": Base64Decoder()
    }

    result = DisallowEncoderOnFinalStepRule.validate(chain_context)
    assert result is None