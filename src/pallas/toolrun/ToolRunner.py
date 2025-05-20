import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pallas.tools.Tool import Tool, ToolError
from pallas.toolchain.ToolDiscovery import ToolDiscovery

class ToolRunner:
    """Class responsible for executing tool chains from a file."""

    def __init__(self, toolchains_file: str, input_text: str, tools_dir: Optional[Path] = None, verbose: bool = False):
        """Initialize the tool runner.

        Args:
            toolchains_file: Path to the file containing tool chains to execute.
            input_text: The input text to process through the tool chains.
            tools_dir: Path to the tools directory. If None, uses the default tools directory.
            verbose: Whether to enable verbose logging.
        """
        self.toolchains_file = Path(toolchains_file)
        self.input_text = input_text
        self.tools_dir = tools_dir
        self.verbose = verbose
        self.tools: Dict[str, Tool] = {}
        self.output_file = Path('out') / f'toolrun_{uuid.uuid4()}.txt'
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.stats = {
            'chains_processed': 0,
            'chains_succeeded': 0,
            'chains_failed': 0,
            'tools_loaded': 0
        }

    def _log(self, message: str) -> None:
        """Print message only if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _load_tools(self) -> None:
        """Load all available tools and create a name-to-tool mapping."""
        self._log("Loading tools...")
        discovery = ToolDiscovery(tools_dir=self.tools_dir)
        tools = discovery.discover_tools()
        self.tools = {tool.name: tool for tool in tools}
        self.stats['tools_loaded'] = len(self.tools)
        self._log(f"Loaded {len(self.tools)} tools: {', '.join(self.tools.keys())}")

    def _execute_chain(self, chain: List[str]) -> Tuple[str, Optional[ToolError]]:
        """Execute a single tool chain.

        Args:
            chain: List of tool names in the chain.

        Returns:
            Tuple[str, Optional[ToolError]]: The final output and any error that occurred.
        """
        self._log(f"\nExecuting chain: {' -> '.join(chain)}")
        self._log(f"Initial input: {self.input_text}")

        current_input = self.input_text
        for i, tool_name in enumerate(chain, 1):
            if tool_name not in self.tools:
                error = ToolError(tool_name=tool_name, message=f"Tool not found: {tool_name}")
                self._log(f"Error: {error}")
                return "", error

            self._log(f"Step {i}/{len(chain)}: Running {tool_name}")
            result, error = self.tools[tool_name].run(current_input)

            if error:
                self._log(f"Error in {tool_name}: {error}")
                return "", error

            self._log(f"Output from {tool_name}: {result}")
            current_input = result

        self._log(f"Chain completed successfully. Final output: {current_input}")
        return current_input, None

    def run(self) -> None:
        """Execute all tool chains from the input file."""
        # First pass: load all tools
        self._load_tools()

        # Second pass: execute chains and write to single output file
        self._log(f"\nExecuting chains from {self.toolchains_file}")
        self._log(f"Output will be written to {self.output_file}")

        with open(self.output_file, 'w') as out_f:
            with open(self.toolchains_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    # Parse the chain
                    chain = [tool.strip() for tool in line.split('->')]
                    self.stats['chains_processed'] += 1

                    # Execute the chain
                    try:
                        output, error = self._execute_chain(chain)
                        if error:
                            out_f.write(f"{' -> '.join(chain)} = Error: {error}\n")
                            self.stats['chains_failed'] += 1
                        else:
                            out_f.write(f"{' -> '.join(chain)} = {output}\n")
                            self.stats['chains_succeeded'] += 1
                    except Exception as e:
                        print(f"Error executing chain {line}: {e}")
                        self.stats['chains_failed'] += 1
                        continue

        if self.verbose:
            self._log("\nToolRunnerExecution Statistics:")
            self._log(f"Tools loaded: {self.stats['tools_loaded']}")
            self._log(f"Chains processed: {self.stats['chains_processed']}")
            self._log(f"Chains succeeded: {self.stats['chains_succeeded']}")
            self._log(f"Chains failed: {self.stats['chains_failed']}")
