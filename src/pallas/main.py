import argparse
import sys
from pathlib import Path
from typing import Optional
import uuid

from pallas.toolchain.ToolChainer import ToolChainer
from pallas.toolchain.ToolProvider import ToolProvider
from pallas.toolrun.ToolRunner import ToolRunner
from pallas.toolchain.rules.rule_map import get_available_rules, get_rule_help, rules as rule_map
from pallas.tools.tool_map import get_available_tools, get_tool_help, tools as tool_map
from pallas.toolchain.rules.RuleEnforcer import RuleEnforcer

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
    parser.add_argument('--rules', nargs='+', choices=get_available_rules(),
                       help=f'Rules to apply to chains. Available rules:\n{get_rule_help()}\n')
    parser.add_argument('--tools', nargs='+', choices=get_available_tools(),
                       help=f'Tools to use in chains. Available tools:\n{get_tool_help()}\n')

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

def create_rule_enforcer(rule_names: Optional[list[str]] = None) -> RuleEnforcer:
    """Create a RuleEnforcer with the specified rules.

    Args:
        rule_names: Optional list of rule names to include.

    Returns:
        RuleEnforcer instance configured with the specified rules.
    """
    if not rule_names:
        return RuleEnforcer([])
    return RuleEnforcer([rule_map[rule] for rule in rule_names])

def run_full_workflow(input_text: str, length: int, verbose: bool, rules: list[str] = None, tool_names: list[str] = None) -> None:
    """Run the full workflow: generate chains and execute them.

    Args:
        input_text: The input text to process through the chains.
        length: Maximum length of tool chains to generate.
        verbose: Whether to enable verbose logging.
        rules: List of rule names to apply.
        tool_names: Optional list of tool names to use. If None, uses all available tools.
    """
    # Generate a UUID for this run
    run_id = str(uuid.uuid4())

    # Create tool discovery and rule enforcer
    tool_provider = ToolProvider(tool_names=tool_names)
    rule_enforcer = create_rule_enforcer(rules)

    # Generate tool chains
    chainer = ToolChainer(tool_provider=tool_provider, max_tree_size=length, verbose=verbose, rule_enforcer=rule_enforcer)
    toolchains_dir = chainer.generate_chains(run_id=run_id)

    # Execute the chains
    runner = ToolRunner(
        toolchains_file=toolchains_dir / f'toolchain_{run_id}.txt',
        input_text=input_text,
        tool_provider=tool_provider,
        verbose=verbose,
        output_filename=f'toolrun_{run_id}.txt'
    )
    runner.run()

def run_tool_chains(toolchains_file: str, input_text: str, verbose: bool, tool_names: list[str] = None) -> None:
    """Run tool chains from a file.

    Args:
        toolchains_file: Path to the file containing tool chains.
        input_text: The input text to process.
        verbose: Whether to enable verbose output.
        tool_names: Optional list of tool names to use. If None, uses all available tools.
    """
    tool_provider = ToolProvider(tool_names=tool_names)
    runner = ToolRunner(toolchains_file, input_text, tool_provider=tool_provider, verbose=verbose)
    runner.run()

def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.all:
        run_full_workflow(args.all, args.length, args.verbose, args.rules, args.tools)
    elif args.run:
        run_tool_chains(args.run, args.input, args.verbose, args.tools)
    else:
        # Generate tool chains
        tool_provider = ToolProvider(tool_names=args.tools)
        rule_enforcer = create_rule_enforcer(args.rules)

        chainer = ToolChainer(tool_provider=tool_provider, max_tree_size=args.length, verbose=args.verbose, rule_enforcer=rule_enforcer)
        run_id = str(uuid.uuid4())
        chainer.generate_chains(run_id=run_id)

if __name__ == '__main__':
    main()