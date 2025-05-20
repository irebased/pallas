from pallas.tools.Tool import Tool

class MockTool(Tool):
    def __init__(self, name: str, domain_chars: set, range_chars: set):
        self.name = name
        self.domain_chars = domain_chars
        self.range_chars = range_chars

    def _process(self, input_str: str) -> str:
        """Mock implementation of _process that returns the input unchanged."""
        return input_str