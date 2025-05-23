import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from pallas.tools.Tool import Tool, ToolError
from pallas.toolchain.ToolProvider import ToolProvider
from pallas.utils.logging_helpers import LoggingHelper
from pallas.utils.chain_utils import format_chain

class ToolRunner:
    """Class responsible for executing tool chains from a file."""

    def __init__(self, toolchains_file: str, input_text: str, tool_provider: ToolProvider,
                 verbose: bool = False, output_filename: Optional[str] = None):
        """Initialize the tool runner.

        Args:
            toolchains_file: Path to the file containing tool chains to execute.
            input_text: The input text to process through the tool chains.
            tool_provider: ToolProvider instance to use for loading tools.
            verbose: Whether to enable verbose logging.
            output_filename: Optional filename for the output file. If None, uses 'toolrun.txt'.
        """
        self.toolchains_file = Path(toolchains_file)
        self.input_text = input_text
        self.tool_provider = tool_provider
        self.verbose = verbose
        self.tools: Dict[str, Tool] = {}
        self.run_id = str(uuid.uuid4())
        self.output_dir = Path('out')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {
            'chains_processed': 0,
            'chains_succeeded': 0,
            'chains_failed': 0,
            'tools_loaded': 0
        }
        self.logger = LoggingHelper(__name__, verbose, self.run_id)

    def _load_tools(self) -> None:
        """Load all available tools and create a name-to-tool mapping."""
        self.logger.log("Loading tools...")
        tools = self.tool_provider.discover_tools()
        self.tools = {tool.name: tool for tool in tools}
        self.stats['tools_loaded'] = len(self.tools)
        self.logger.log(f"Loaded {len(self.tools)} tools: {', '.join(self.tools.keys())}")

    def _execute_chain(self, chain: List[str]) -> Tuple[str, Optional[ToolError]]:
        """Execute a single tool chain.

        Args:
            chain: List of tool names in the chain.

        Returns:
            Tuple[str, Optional[ToolError]]: The final output and any error that occurred.
        """
        self.logger.log(f"\nExecuting chain: {' -> '.join(chain)}")
        self.logger.log(f"Initial input: {self.input_text}")

        current_input = self.input_text
        for i, tool_name in enumerate(chain, 1):
            if tool_name not in self.tools:
                error = ToolError(tool_name, f"Tool not found: {tool_name}")
                self.logger.log_error(f"Error: {error}")
                return "", error

            self.logger.log(f"Step {i}/{len(chain)}: Running {tool_name}")
            result, sep, error = self.tools[tool_name].run(current_input)

            if error:
                self.logger.log_error(f"Error in {tool_name}: {error}")
                return "", ToolError(tool_name, error.message)

            self.logger.log(f"Output from {tool_name}: {result}")
            current_input = result

        self.logger.log(f"Chain completed successfully. Final output: {current_input}")
        return current_input, None

    def run(self) -> None:
        """Execute all tool chains from the input file."""
        # First pass: load all tools
        self._load_tools()

        # Second pass: execute chains and write to separate output files
        self.logger.log(f"\nExecuting chains from {self.toolchains_file}")

        success_file = self.output_dir / f'toolrun_succeeded_{self.run_id}.txt'
        failed_file = self.output_dir / f'toolrun_failed_{self.run_id}.txt'

        self.logger.log(f"Successful chains will be written to {success_file}")
        self.logger.log(f"Failed chains will be written to {failed_file}")

        with open(success_file, 'w') as success_f, open(failed_file, 'w') as failed_f:
            with open(self.toolchains_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    # Parse the chain
                    chain = [tool.strip() for tool in line.split('->')]
                    chain_str = format_chain(chain)
                    self.stats['chains_processed'] += 1

                    # Execute the chain
                    try:
                        output, error = self._execute_chain(chain)
                        if error:
                            failed_f.write(f"{chain_str} = Error: {error}\n")
                            self.stats['chains_failed'] += 1
                        else:
                            success_f.write(f"{chain_str} = {output}\n")
                            self.stats['chains_succeeded'] += 1
                    except Exception as e:
                        failed_f.write(f"{chain_str} = Error: {str(e)}\n")
                        self.stats['chains_failed'] += 1
                        continue

        if self.verbose:
            self.logger.log("\nToolRunnerExecution Statistics:")
            self.logger.log(f"Tools loaded: {self.stats['tools_loaded']}")
            self.logger.log(f"Chains processed: {self.stats['chains_processed']}")
            self.logger.log(f"Chains succeeded: {self.stats['chains_succeeded']}")
            self.logger.log(f"Chains failed: {self.stats['chains_failed']}")
