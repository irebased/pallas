import time
from typing import List, Set, Optional
from pallas.tools.Tool import Tool
from pathlib import Path
from pallas.toolchain.ToolProvider import ToolProvider
from pallas.toolchain.ChainContext import ChainContext
from pallas.toolchain.rules.RuleEnforcer import RuleEnforcer
from pallas.utils.tree_utils import calculate_max_tree_size

class ToolChainer:
    """Class responsible for generating valid tool chains."""

    def __init__(self, tool_provider: ToolProvider, max_tree_size: int = 3,
                 output_filename: Optional[str] = None, verbose: bool = False,
                 rule_enforcer: Optional['RuleEnforcer'] = None):
        """Initialize the tool chainer.

        Args:
            tool_provider: ToolDiscovery instance to use for loading tools.
            max_tree_size: Maximum number of tools in a chain.
            output_filename: Optional filename for the output file. If None, uses 'toolchain.txt'.
            verbose: Whether to enable verbose logging.
            rule_enforcer: Optional RuleEnforcer instance to use for chain validation.
        """
        self.tool_provider = tool_provider
        self.max_tree_size = max_tree_size
        self.verbose = verbose
        self.tools: List[Tool] = []
        self.valid_chains: List[List[str]] = []
        self.visited_nodes: int = 0
        self.output_file = Path('out') / (output_filename or 'toolchain.txt')
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.pruned_chains: List[List[Tool]] = []
        self.phase_times = {}
        self.rule_enforcer = rule_enforcer or RuleEnforcer([])

    def generate_chains(self, run_id: Optional[str] = None) -> Path:
        """Generate all valid tool chains and write them to a file.

        Args:
            run_id: Optional UUID to use in the output filename. If None, uses 'toolchain.txt'.

        Returns:
            Path: The output directory path.
        """
        # Reset state
        self.valid_chains = []
        self.visited_nodes = 0

        self._load_tools()

        available_tools = set(range(len(self.tools)))
        self._generate_chains([], available_tools)

        if run_id:
            self.output_file = self.output_file.parent / f'toolchain_{run_id}.txt'

        with open(self.output_file, 'w') as f:
            for chain in self.valid_chains:
                f.write(' -> '.join(self.tools[i].name for i in chain) + '\n')

        if self.verbose:
            max_possible_nodes = calculate_max_tree_size(self.tools, self.max_tree_size)
            self._log(self.rule_enforcer.format_stats(max_possible_nodes, self.visited_nodes))

        return self.output_file.parent

    def _generate_chains(self, current_chain: List[str], available_tools: Set[str]) -> None:
        """Recursively generate valid tool chains.

        Args:
            current_chain: Current chain being built.
            available_tools: Set of tool names available for the next step.
        """
        self.visited_nodes += 1

        if len(current_chain) >= self.max_tree_size:
            return

        for tool_name in available_tools:
            # Check if the next tool follows all rules
            if not self._is_valid_next_tool(current_chain, tool_name):
                continue

            new_chain = current_chain + [tool_name]

            # Only add to valid_chains if at max_tree_size
            if len(new_chain) == self.max_tree_size:
                self.valid_chains.append(new_chain)
            else:
                self._generate_chains(new_chain, available_tools - {tool_name})

    def _is_valid_next_tool(self, current_chain: List[str], next_tool: str) -> bool:
        """Check if the next tool is valid according to all rules.

        Args:
            current_chain: Current chain being built.
            next_tool: Next tool to be added.

        Returns:
            bool: True if the next tool is valid, False otherwise.
        """
        context = ChainContext(
            current_chain=current_chain,
            next_tool=next_tool,
            target_length=self.max_tree_size,
            tools=self.tools
        )

        if error := self.rule_enforcer.validate_chain_against_rules(context):
            if self.verbose:
                self._log(f"Rule violation: {error.message}")
            return False

        return True

    def _load_tools(self) -> None:
        """Load all available tools."""
        self.tools = self.tool_provider.discover_tools()

    def _log(self, message: str) -> None:
        """Print message only if verbose mode is enabled."""
        if self.verbose:
            print(message)