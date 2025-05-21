from pallas.toolchain.rules.AlternatingRule import AlternatingRule
from pallas.toolchain.rules.BalancingRule import BalancingRule
from pallas.toolchain.rules.CharacterSetRule import CharacterSetRule
from pallas.toolchain.rules.RedundantPairRule import RedundantPairRule

# Map of rule names to their classes
rules = {
    'alternating': AlternatingRule,
    'balancing': BalancingRule,
    'charset': CharacterSetRule,
    'redundant': RedundantPairRule,
}

# Help text for each rule
rule_help = {
    'alternating': "Enforce alternating encoder/decoder operations.",
    'balancing': "Ensure balanced numbers of encoders and decoders.",
    'charset': "Ensure compatible character sets between adjacent tools.",
    'redundant': "Prevent redundant encode-decode operations.",
}

def get_available_rules() -> list[str]:
    """Get a list of available rule names."""
    return list(rules.keys())

def get_rule_help() -> str:
    """Get formatted help text for all available rules."""
    return "\n".join(f"  {name}: {help_text}" for name, help_text in rule_help.items())