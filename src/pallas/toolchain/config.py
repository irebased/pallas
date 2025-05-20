"""
Configuration for tool discovery and chaining.
"""

# Files to exclude from tool discovery
EXCLUDED_FILES = {
    '__init__.py',
    'Tool.py',
    'ToolError.py',
    'config.py',
    'ToolChainer.py',
}

# Classes to exclude from tool discovery
EXCLUDED_CLASSES = {
    'Tool',
    'ToolError',
}