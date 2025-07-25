import argparse
import sys
from pathlib import Path
from typing import Optional
import uuid
import json

from pallas.toolchain.ToolChainer import ToolChainer
from pallas.toolchain.ToolProvider import ToolProvider
from pallas.toolrun.ToolRunner import ToolRunner
from pallas.toolchain.rules.rule_map import get_available_rules, get_rule_help, rules as rule_map
from pallas.tools.tool_map import get_available_tools, get_tool_help, tools as tool_map
from pallas.toolchain.rules.RuleEnforcer import RuleEnforcer

# Argument schema for both CLI and JSON
ARGUMENTS = [
    {
        "flags": ["-l", "--length"],
        "dest": "length",
        "kwargs": {"type": int, "help": "Length of tool chains to generate (default 3)", "default": None}
    },
    {
        "flags": ["-v", "--verbose"],
        "dest": "verbose",
        "kwargs": {"action": "store_true", "help": "Enable verbose output for logging statistics", "default": False}
    },
    {
        "flags": ["-b", "--balance-encodings"],
        "dest": "balance_encodings",
        "kwargs": {"action": "store_true", "help": "Balance encode/decode operations in chains", "default": False}
    },
    {
        "flags": ["-s", "--strict-alternating"],
        "dest": "strict_alternating",
        "kwargs": {"action": "store_true", "help": "Enforce strictly alternating encoder/decoder operations", "default": False}
    },
    {
        "flags": ["--rules"],
        "dest": "rules",
        "kwargs": {"nargs": "+", "choices": get_available_rules(), "help": f"Rules to apply to chains. Available rules:\n{get_rule_help()}\n", "default": None}
    },
    {
        "flags": ["--tools"],
        "dest": "tools",
        "kwargs": {"nargs": "+", "choices": get_available_tools(), "help": f"Tools to use in chains. Available tools:\n{get_tool_help()}\n", "default": None}
    },
    {
        "flags": ["-r", "--run"],
        "dest": "run",
        "kwargs": {"type": str, "help": "Run tool chains from a provided toolchain output file", "default": None}
    },
    {
        "flags": ["-i", "--input"],
        "dest": "input",
        "kwargs": {"type": str, "help": "Input text to process through tool chain runs", "default": None}
    },
    {
        "flags": ["-a", "--all"],
        "dest": "all",
        "kwargs": {"type": str, "help": "Run full workflow with input text. Specify --length (default 3)", "default": None}
    },
    {
        "flags": ["--json-input"],
        "dest": "json_input",
        "kwargs": {"type": str, "help": "Path to JSON file with all arguments", "default": None}
    },
]

def build_parser():
    parser = argparse.ArgumentParser(description='Pallas Tool Chain Generator and Runner')
    for arg in ARGUMENTS:
        parser.add_argument(*arg["flags"], dest=arg["dest"], **arg["kwargs"])
    return parser

def parse_args() -> argparse.Namespace:
    """Parse command line arguments, supporting both CLI and JSON input."""
    parser = build_parser()
    args = parser.parse_args()
    args_dict = vars(args)

    # If --json-input is provided, load arguments from JSON and merge
    if args.json_input:
        json_path = Path(args.json_input)
        if not json_path.exists():
            parser.error(f"JSON input file not found: {json_path}")
        try:
            with open(json_path, 'r') as f:
                json_args = json.load(f)
        except Exception as e:
            parser.error(f"Failed to parse JSON input: {e}")

        # Merge: CLI args take precedence if set (not None or not False for bools)
        for arg in ARGUMENTS:
            dest = arg["dest"]
            if dest in json_args:
                cli_val = args_dict.get(dest)
                # For bools, only override if CLI is default (False) and JSON is True
                if arg["kwargs"].get("action") == "store_true":
                    if cli_val is False:
                        setattr(args, dest, json_args[dest])
                # For others, override if CLI is None
                elif cli_val is None:
                    setattr(args, dest, json_args[dest])

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