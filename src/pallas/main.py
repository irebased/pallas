import argparse
import sys
from pathlib import Path
from typing import Optional
import uuid

from pallas.toolchain.ToolChainer import ToolChainer
from pallas.toolchain.ToolDiscovery import ToolDiscovery
from pallas.toolrun.ToolRunner import ToolRunner

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pallas Tool Chain Generator and Runner')

    # Tool chain generation options
    parser.add_argument('-l', '--length', type=int, help='Length of tool chains to generate (default 3)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output for logging statistics')
    parser.add_argument('-b', '--balance-encodings', action='store_true',
                       help='Balance encode/decode operations in chains')
    parser.add_argument('-s', '--strict-alternating', action='store_true',
                       help='Enforce strictly alternating encoder/decoder operations')

    # Tool chain running options
    parser.add_argument('-r', '--run', type=str, help='Run tool chains from a provided toolchain output file')
    parser.add_argument('-i', '--input', type=str, help='Input text to process through tool chain runs')

    # Full workflow option
    parser.add_argument('-a', '--all', type=str, help='Run full workflow with input text. Specify --length (default 3)')

    args = parser.parse_args()

    # Validate argument combinations
    if args.run and not args.input:
        parser.error("--input is required when using --run")
    if args.all and (args.run or args.input):
        parser.error("--all cannot be used with --run or --input")
    if args.all and not args.length:
        parser.error("--length is required when using --all")
    if not args.all and not args.run and not args.length:
        parser.error("--length is required for chain generation")

    return args

def run_full_workflow(input_text: str, length: int, verbose: bool) -> None:
    """Run the full workflow: generate chains and execute them.

    Args:
        input_text: The input text to process through the chains.
        length: Maximum length of tool chains to generate.
        verbose: Whether to enable verbose logging.
    """
    # Generate a UUID for this run
    run_id = str(uuid.uuid4())

    # Load tools
    discovery = ToolDiscovery()
    tools = discovery.discover_tools()

    # Generate tool chains
    chainer = ToolChainer(max_tree_size=length, verbose=verbose)
    toolchains_dir = chainer.generate_chains(run_id=run_id)

    # Execute the chains
    runner = ToolRunner(
        toolchains_file=toolchains_dir / f'toolchain_{run_id}.txt',
        input_text=input_text,
        verbose=verbose,
        output_filename=f'toolrun_{run_id}.txt'
    )
    runner.run()

def run_tool_chains(toolchains_file: str, input_text: str, verbose: bool) -> None:
    """Run tool chains from a file.

    Args:
        toolchains_file: Path to the file containing tool chains.
        input_text: The input text to process.
        verbose: Whether to enable verbose output.
    """
    runner = ToolRunner(toolchains_file, input_text, verbose=verbose)
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

        chainer = ToolChainer(max_tree_size=args.length, verbose=args.verbose,
                            balance_encodings=args.balance_encodings,
                            strict_alternating=args.strict_alternating)
        run_id = str(uuid.uuid4())
        chainer.generate_chains(run_id=run_id)

if __name__ == '__main__':
    main()