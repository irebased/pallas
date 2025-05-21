import pytest
from pallas.utils.tree_utils import calculate_max_tree_size

def test_calculate_max_tree_size_empty_tools():
    """Test calculation with empty tools list."""
    assert calculate_max_tree_size([], 3) == 0

def test_calculate_max_tree_size_single_tool():
    """Test calculation with a single tool."""
    assert calculate_max_tree_size(['tool1'], 3) == 3

def test_calculate_max_tree_size_two_tools():
    """Test calculation with two tools."""
    # For 2 tools and max_tree_size=3:
    # Level 1: 2 nodes
    # Level 2: 4 nodes (2²)
    # Level 3: 8 nodes (2³)
    # Total = 2 + 4 + 8 = 14
    assert calculate_max_tree_size(['tool1', 'tool2'], 3) == 14

def test_calculate_max_tree_size_three_tools():
    """Test calculation with three tools."""
    # For 3 tools and max_tree_size=2:
    # Level 1: 3 nodes
    # Level 2: 9 nodes (3²)
    # Total = 3 + 9 = 12
    assert calculate_max_tree_size(['tool1', 'tool2', 'tool3'], 2) == 12

def test_calculate_max_tree_size_large_input():
    """Test calculation with larger number of tools."""
    # For 5 tools and max_tree_size=2:
    # Level 1: 5 nodes
    # Level 2: 25 nodes (5²)
    # Total = 5 + 25 = 30
    assert calculate_max_tree_size(['tool1', 'tool2', 'tool3', 'tool4', 'tool5'], 2) == 30

def test_calculate_max_tree_size_max_depth_one():
    """Test calculation with max_tree_size=1."""
    tools = ['tool1', 'tool2', 'tool3']
    # For max_tree_size=1, we only have the first level
    assert calculate_max_tree_size(tools, 1) == len(tools)

def test_calculate_max_tree_size_zero_max_depth():
    """Test calculation with max_tree_size=0."""
    tools = ['tool1', 'tool2', 'tool3']
    assert calculate_max_tree_size(tools, 0) == 0