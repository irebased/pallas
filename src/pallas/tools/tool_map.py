from pallas.tools.encoders.octal import OctalEncoder
from pallas.tools.encoders.decimal import DecimalEncoder
from pallas.tools.encoders.base64 import Base64Encoder
from pallas.tools.encoders.hex import HexEncoder
from pallas.tools.decoders.octal import OctalDecoder
from pallas.tools.decoders.decimal import DecimalDecoder
from pallas.tools.decoders.base64 import Base64Decoder
from pallas.tools.decoders.hex import HexDecoder

# Map of tool names to their classes
tools = {
    'octal_encoder': OctalEncoder,
    'decimal_encoder': DecimalEncoder,
    'base64_encoder': Base64Encoder,
    'hex_encoder': HexEncoder,
    'octal_decoder': OctalDecoder,
    'decimal_decoder': DecimalDecoder,
    'base64_decoder': Base64Decoder,
    'hex_decoder': HexDecoder,
}

# Help text for each tool
tool_help = {
    'octal_encoder': "Converts ASCII text to octal representation.",
    'decimal_encoder': "Converts ASCII text to decimal representation.",
    'base64_encoder': "Converts ASCII text to Base64 encoding.",
    'hex_encoder': "Converts ASCII text to hexadecimal representation.",
    'octal_decoder': "Converts octal representation back to ASCII text.",
    'decimal_decoder': "Converts decimal representation back to ASCII text.",
    'base64_decoder': "Converts Base64 encoding back to ASCII text.",
    'hex_decoder': "Converts hexadecimal representation back to ASCII text.",
}

def get_available_tools() -> list[str]:
    """Get a list of available tool names."""
    return list(tools.keys())

def get_tool_help() -> str:
    """Get formatted help text for all available tools."""
    return "\n".join(f"  {name}: {help_text}" for name, help_text in tool_help.items())