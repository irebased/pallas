import argparse
import sys
from pathlib import Path
from typing import Optional

from pallas.toolchain.ToolChainer import ToolChainer
from pallas.toolchain.ToolDiscovery import ToolDiscovery
from pallas.toolrun.ToolRunner import ToolRunner

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pallas Tool Chain Generator and Runner')

    # Tool chain generation options
    parser.add_argument('-l', '--length', type=int, help='Length of tool chains to generate (default 3)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output for logging statistics')
    parser.add_argument('-o', '--output', type=str, help='Output file for generated chains')

    # Tool chain running options
    parser.add_argument('-r', '--run', type=str, help='Run tool chains from a provided toolchain output file')
    parser.add_argument('-i', '--input', type=str, help='Input text to process through tool chain runs')

    # Full workflow option
    parser.add_argument('-a', '--all', type=str, help='Run full workflow with input text. Specify --length (default 3)')

    args = parser.parse_args()

    # Validate argument combinations
    if args.run and not args.input:
        parser.error("--input is required when using --run")
    if args.all and (args.run or args.input or args.output):
        parser.error("--all cannot be used with --run, --input, or --output")
    if args.all and not args.length:
        parser.error("--length is required when using --all")
    if not args.all and not args.run and not args.length:
        parser.error("--length is required for chain generation")

    return args

def run_tool_chains(toolchains_file: str, input_text: str, verbose: bool = False) -> None:
    """Run tool chains from a file."""
    runner = ToolRunner(toolchains_file, input_text, verbose=verbose)
    runner.run()

def run_full_workflow(input_text: str, chain_length: int, verbose: bool = False) -> None:
    """Run the full workflow: generate chains and execute them.

    Args:
        input_text: The input text to process through the chains.
        chain_length: The length of chains to generate.
        verbose: Whether to enable verbose output.
    """
    # Discover tools
    discovery = ToolDiscovery()
    tools = discovery.discover_tools()

    # Generate chains
    chainer = ToolChainer(chain_length=chain_length, tools=tools, verbose=verbose)
    output_dir = chainer.generate_chains('out/toolchains.txt')

    # Run the generated chains
    runner = ToolRunner(output_dir / 'toolchains.txt', input_text, verbose=verbose)
    runner.run()

def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.all:
        run_full_workflow(args.all, args.length, args.verbose)
    elif args.run:
        run_tool_chains(args.run, args.input, args.verbose)
    else:
        # Generate tool chains
        discovery = ToolDiscovery()
        tools = discovery.discover_tools()

        chainer = ToolChainer(chain_length=args.length, tools=tools, verbose=args.verbose)
        output_file = args.output or 'out/toolchains.txt'
        chainer.generate_chains(output_file)

if __name__ == '__main__':
    main()