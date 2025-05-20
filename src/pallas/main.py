import os
import sys
import argparse
from pathlib import Path
from pallas.toolchain.ToolChainer import ToolChainer
from pallas.toolchain.ToolDiscovery import ToolDiscovery
from pallas.toolchain.config import EXCLUDED_FILES, EXCLUDED_CLASSES
def generate_chains(chain_length: int = 3, verbose: bool = False):
    """Generate tool chains and save them to a file."""
    output_dir = Path("out")
    output_dir.mkdir(exist_ok=True)

    tools_dir = Path(__file__).parent / 'tools'
    discovery = ToolDiscovery(tools_dir=tools_dir, excluded_files=EXCLUDED_FILES, excluded_classes=EXCLUDED_CLASSES)
    tools = discovery.discover_tools()

    chainer = ToolChainer(chain_length=chain_length, tools=tools, verbose=verbose)
    chainer.generate_chains("out/toolchains.txt")

def main():
    parser = argparse.ArgumentParser(
        description="Pallas Tool Chain Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pallas.main --generate-chain  # Generate chains with default length
  python -m pallas.main -c -l 4          # Generate chains with length 4
  python -m pallas.main -c -v            # Generate chains with verbose output
        """
    )

    parser.add_argument(
        "-c", "--generate-chain",
        action="store_true",
        help="Generate tool chains"
    )

    parser.add_argument(
        "-l", "--length",
        type=int,
        default=3,
        help="Length of tool chains to generate (default: 3)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(1)

    if args.generate_chain:
        generate_chains(chain_length=args.length, verbose=args.verbose)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()