from pallas.toolchain.rules.AlternatingRule import AlternatingRule
from pallas.toolchain.rules.BalancingEncoderDecoderRule import BalancingEncoderDecoderRule
from pallas.toolchain.rules.CharacterSetRule import CharacterSetRule
from pallas.toolchain.rules.RedundantPairRule import RedundantPairRule
from pallas.toolchain.rules.DisallowEncoderOnFinalStep import DisallowEncoderOnFinalStepRule

# Map of rule names to their classes
rules = {
    'alternating': AlternatingRule,
    'balancing': BalancingEncoderDecoderRule,
    'charset': CharacterSetRule,
    'redundant': RedundantPairRule,
    'no_encoder_ending': DisallowEncoderOnFinalStepRule,
}

# Help text for each rule
rule_help = {
    'alternating': "Enforce alternating encoder/decoder operations.",
    'balancing': "Ensure balanced numbers of encoders and decoders.",
    'charset': "Ensure compatible character sets between adjacent tools.",
    'redundant': "Prevent redundant encode-decode operations.",
    'no_encoder_ending': "Prevent encoders from ending the chain.",
}

def get_available_rules() -> list[str]:
    """Get a list of available rule names."""
    return list(rules.keys())

def get_rule_help() -> str:
    """Get formatted help text for all available rules."""
    return "\n".join(f"  {name}: {help_text}" for name, help_text in rule_help.items())