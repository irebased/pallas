from typing import List

def calculate_max_tree_size(tools: List[str], max_tree_size: int) -> int:
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
        n = len(tools)
        if n == 0:
            return 0
        if n == 1:
            return max_tree_size

        # For a tree of depth L where each node has N children:
        # Total nodes = N + N² + N³ + ... + N^L
        # This is a geometric series with first term a=N and ratio r=N
        # Sum = N(1 - N^L)/(1 - N)
        return int(n * (1 - n**max_tree_size) / (1 - n))