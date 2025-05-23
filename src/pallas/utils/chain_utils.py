from typing import List

def format_chain(chain: List[str]) -> str:
    """Format a chain for display.

    Args:
        chain: List of tool names in the chain.

    Returns:
        str: Formatted chain string.
    """
    if len(chain) < 2:
        return " = ".join(chain)
    return " -> ".join(chain[:-1] + [" = ".join(chain[-1:])])