import time
from typing import List, Set, Dict
from pallas.tools.Tool import Tool
from pathlib import Path

class ToolChainer:
    def __init__(self, chain_length: int, tools: List[Tool], verbose: bool = False):
        self.chain_length = chain_length
        self.verbose = verbose
        self.tools = tools
        self.invalid_connections: Dict[str, Set[str]] = {}  # tool_name -> set of invalid next tools
        self.valid_chains: List[List[Tool]] = []
        self.pruned_chains: List[List[Tool]] = []
        self.visited_nodes = 0
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
            return self.chain_length

        # For a tree of depth L where each node has N children:
        # Total nodes = N + N² + N³ + ... + N^L
        # This is a geometric series with first term a=N and ratio r=N
        # Sum = N(1 - N^L)/(1 - N)
        return int(n * (1 - n**self.chain_length) / (1 - n))

    def _is_redundant_encode_decode(self, tool_a: Tool, tool_b: Tool) -> bool:
        """
        Check if the connection between tools represents a redundant encode-decode pair.
        Returns True if the connection should be pruned.
        """
        def get_base_name(tool_name: str) -> str:
            return tool_name.rsplit('_', 1)[0]

        def get_operation(tool_name: str) -> str:
            return tool_name.split('_')[-1]

        base_a = get_base_name(tool_a.name)
        base_b = get_base_name(tool_b.name)
        op_a = get_operation(tool_a.name)
        op_b = get_operation(tool_b.name)

        if base_a == base_b:
            # Prune encode -> decode and decode -> encode for same base
            if (op_a == 'encoder' and op_b == 'decoder') or \
               (op_a == 'decoder' and op_b == 'encoder'):
                self.pruning_stats['redundant_pairs'] += 1
                self._log(f"Pruning redundant {base_a} encode-decode pair: {tool_a.name} -> {tool_b.name}")
                return True

        return False

    def _is_valid_connection(self, tool_a: Tool, tool_b: Tool) -> bool:
        """
        Check if tool_a's range_chars are compatible with tool_b's domain_chars.
        A connection is valid if there is any overlap in the character sets,
        as tools handle invalid characters through error handling.
        """
        if tool_a.name in self.invalid_connections and tool_b.name in self.invalid_connections[tool_a.name]:
            self.pruning_stats['memoized'] += 1
            self._log(f"Using memoized invalid connection: {tool_a.name} -> {tool_b.name}")
            return False

        if self._is_redundant_encode_decode(tool_a, tool_b):
            return False

        range_chars = tool_a.range_chars
        domain_chars = tool_b.domain_chars

        is_valid = (range_chars.issubset(domain_chars) or
                   range_chars.issuperset(domain_chars) or
                   range_chars == domain_chars)

        if not is_valid:
            if tool_a.name not in self.invalid_connections:
                self.invalid_connections[tool_a.name] = set()
            self.invalid_connections[tool_a.name].add(tool_b.name)
            self.pruning_stats['char_set_mismatch'] += 1
            self._log(f"Pruning due to character set mismatch: {tool_a.name} -> {tool_b.name}")
            self._log(f"  {tool_a.name} range_chars: {sorted(range_chars)}")
            self._log(f"  {tool_b.name} domain_chars: {sorted(domain_chars)}")

        return is_valid

    def _dfs(self, current_chain: List[Tool], current_tools: List[Tool]) -> None:
        """Perform DFS to find valid tool chains."""
        self.visited_nodes += 1

        if len(current_chain) == self.chain_length:
            self.valid_chains.append(current_chain.copy())
            self._log(f"Found valid chain: {' -> '.join(tool.name for tool in current_chain)}")
            return

        last_tool = current_tools[-1] if current_tools else None
        for tool in self.tools:
            if tool not in current_chain:
                if last_tool is None or self._is_valid_connection(last_tool, tool):
                    current_chain.append(tool)
                    current_tools.append(tool)
                    self._dfs(current_chain, current_tools)
                    current_chain.pop()
                    current_tools.pop()
                else:
                    pruned_chain = current_chain + [tool]
                    self.pruned_chains.append(pruned_chain)
                    self._log(f"Pruned invalid chain: {' -> '.join(tool.name for tool in pruned_chain)}")

    def generate_chains(self, output_file: str) -> Path:
        """Generate all valid tool chains and write them to a file.

        Args:
            output_file: Path to the output file.

        Returns:
            Path: The directory containing the output file.
        """
        # Reset state
        self.valid_chains = []
        self.pruned_chains = []
        self.visited_nodes = 0
        self.pruning_stats = {
            'redundant_pairs': 0,
            'char_set_mismatch': 0,
            'memoized': 0
        }
        self.phase_times = {}

        # Calculate maximum tree size
        max_size = self._calculate_max_tree_size()
        if self.verbose:
            self._log(f"Maximum tree size: {max_size}")

        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate chains using DFS
        start_time = time.time()
        self._dfs([], [])
        self.phase_times['dfs'] = time.time() - start_time

        # Write chains to file
        with open(output_file, 'w') as f:
            for chain in self.valid_chains:
                f.write(f"{' -> '.join(tool.name for tool in chain)}\n")

        if self.verbose:
            self._log(f"Generated {len(self.valid_chains)} valid chains")
            self._log(f"Pruned {len(self.pruned_chains)} chains")
            self._log(f"Visited {self.visited_nodes} nodes")
            self._log("Pruning statistics:")
            for stat, count in self.pruning_stats.items():
                self._log(f"  {stat}: {count}")
            self._log("Phase times:")
            for phase, duration in self.phase_times.items():
                self._log(f"  {phase}: {duration:.2f}s")

        return output_path.parent