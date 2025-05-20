import time
import os
import uuid
from typing import List, Set, Dict, Optional, Tuple
from pallas.tools.Tool import Tool, ToolError
from pathlib import Path
from pallas.toolchain.ToolDiscovery import ToolDiscovery

class ToolChainer:
    """Class responsible for generating valid tool chains."""

    def __init__(self, tools_dir: Optional[Path] = None, max_tree_size: int = 3,
                 output_filename: Optional[str] = None, verbose: bool = False):
        """Initialize the tool chainer.

        Args:
            tools_dir: Path to the tools directory. If None, uses the default tools directory.
            max_tree_size: Maximum number of tools in a chain.
            output_filename: Optional filename for the output file. If None, uses 'toolchain.txt'.
            verbose: Whether to enable verbose logging.
        """
        self.tools_dir = tools_dir
        self.max_tree_size = max_tree_size
        self.verbose = verbose
        self.tools: List[Tool] = []
        self.valid_chains: List[List[str]] = []
        self.visited_nodes: int = 0
        self.output_file = Path('out') / (output_filename or 'toolchain.txt')
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.invalid_connections: Dict[str, Set[str]] = {}  # tool_name -> set of invalid next tools
        self.pruned_chains: List[List[Tool]] = []
        self.pruning_stats = {
            'redundant_pairs': 0,
            'char_set_mismatch': 0,
            'memoized': 0
        }
        self.phase_times = {}

    def _log(self, message: str) -> None:
        """Print message only if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _calculate_max_tree_size(self) -> int:
        """
        Calculate the theoretical maximum number of nodes in a complete tree.
        For a chain of length L with N tools:
        - Level 1: N nodes (all tools as starting points)
        - Level 2: N * N nodes (each tool can connect to any tool, including itself)
        - Level 3: N * N * N nodes
        - And so on...
        Total = N + N² + N³ + ... + N^L
        This is a geometric series with first term a=N and ratio r=N
        Sum = N(1 - N^L)/(1 - N)
        """
        n = len(self.tools)
        if n == 0:
            return 0
        if n == 1:
            return self.max_tree_size

        # For a tree of depth L where each node has N children:
        # Total nodes = N + N² + N³ + ... + N^L
        # This is a geometric series with first term a=N and ratio r=N
        # Sum = N(1 - N^L)/(1 - N)
        return int(n * (1 - n**self.max_tree_size) / (1 - n))

    def _load_tools(self) -> None:
        """Load all available tools."""
        discovery = ToolDiscovery(tools_dir=self.tools_dir)
        self.tools = discovery.discover_tools()

    def _is_redundant_encode_decode(self, tool1: Tool, tool2: Tool) -> bool:
        """Check if the connection is a redundant encode-decode operation.

        Args:
            tool1: First tool in the chain.
            tool2: Second tool in the chain.

        Returns:
            bool: True if the connection is redundant, False otherwise.
        """
        # Get base names by removing _encoder or _decoder suffix
        def get_base_name(name: str) -> str:
            if '_encoder' in name:
                return name.replace('_encoder', '')
            if '_decoder' in name:
                return name.replace('_decoder', '')
            return name

        # Get operation type (encode or decode)
        def get_operation(name: str) -> str:
            if '_encoder' in name:
                return 'encode'
            if '_decoder' in name:
                return 'decode'
            return 'unknown'

        base1 = get_base_name(tool1.name)
        base2 = get_base_name(tool2.name)
        op1 = get_operation(tool1.name)
        op2 = get_operation(tool2.name)

        # Check if it's a redundant pair (same base, opposite operations)
        if base1 == base2 and op1 != op2 and op1 != 'unknown' and op2 != 'unknown':
            self.pruning_stats['redundant_pairs'] += 1
            self._log(f"Pruning redundant {base1} encode-decode pair: {tool1.name} -> {tool2.name}")
            return True

        return False

    def _is_valid_connection(self, tool1: Tool, tool2: Tool) -> bool:
        """Check if two tools can be connected in a chain.

        Args:
            tool1: First tool in the chain.
            tool2: Second tool in the chain.

        Returns:
            bool: True if the tools can be connected, False otherwise.
        """
        if tool1.name in self.invalid_connections and tool2.name in self.invalid_connections[tool1.name]:
            self.pruning_stats['memoized'] += 1
            self._log(f"Using memoized invalid connection: {tool1.name} -> {tool2.name}")
            return False

        if self._is_redundant_encode_decode(tool1, tool2):
            return False

        range_chars = tool1.range_chars
        domain_chars = tool2.domain_chars

        is_valid = (range_chars.issubset(domain_chars) or
                   range_chars.issuperset(domain_chars) or
                   range_chars == domain_chars)

        if not is_valid:
            if tool1.name not in self.invalid_connections:
                self.invalid_connections[tool1.name] = set()
            self.invalid_connections[tool1.name].add(tool2.name)
            self.pruning_stats['char_set_mismatch'] += 1
            self._log(f"Pruning due to character set mismatch: {tool1.name} -> {tool2.name}")
            self._log(f"  {tool1.name} range_chars: {sorted(range_chars)}")
            self._log(f"  {tool2.name} domain_chars: {sorted(domain_chars)}")

        return is_valid

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
            # Skip if this would create a redundant encode-decode pair
            if current_chain and self._is_redundant_encode_decode(
                self.tools[current_chain[-1]], self.tools[tool_name]
            ):
                continue

            # Check if the connection is valid
            if not current_chain or self._is_valid_connection(
                self.tools[current_chain[-1]], self.tools[tool_name]
            ):
                new_chain = current_chain + [tool_name]
                self.valid_chains.append(new_chain)
                self._generate_chains(new_chain, available_tools - {tool_name})

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

        # Load tools
        self._load_tools()

        # Generate chains
        available_tools = set(range(len(self.tools)))
        self._generate_chains([], available_tools)

        # Update output filename if run_id is provided
        if run_id:
            self.output_file = self.output_file.parent / f'toolchain_{run_id}.txt'

        # Write chains to file
        with open(self.output_file, 'w') as f:
            for chain in self.valid_chains:
                f.write(' -> '.join(self.tools[i].name for i in chain) + '\n')

        if self.verbose:
            self._log(f"Generated {len(self.valid_chains)} valid chains")
            self._log("Pruning statistics:")
            for stat, count in self.pruning_stats.items():
                self._log(f"  {stat}: {count}")

        return self.output_file.parent