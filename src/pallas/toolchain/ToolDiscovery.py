import os
import importlib
import inspect
import sys
from pathlib import Path
from typing import List, Set, Optional
from pallas.tools.Tool import Tool
from pallas.toolchain.config import EXCLUDED_FILES, EXCLUDED_CLASSES

class ToolDiscovery:
    """Class responsible for discovering and loading tools."""

    def __init__(self, tools_dir: Path = None, excluded_files: Optional[Set[str]] = None, excluded_classes: Optional[Set[str]] = None):
        """Initialize the tool discovery process.

        Args:
            tools_dir: Path to the tools directory. If None, uses the default tools directory.
            excluded_files: Set of file names to exclude from discovery.
            excluded_classes: Set of class names to exclude from discovery.
        """
        self.tools_dir = tools_dir or Path(__file__).parent.parent / 'tools'
        self.excluded_files = excluded_files if excluded_files is not None else EXCLUDED_FILES
        self.excluded_classes = excluded_classes if excluded_classes is not None else EXCLUDED_CLASSES
        self.tools: List[Tool] = []

    def discover_tools(self) -> List[Tool]:
        """Discover all available tools in the tools directory.

        Returns:
            List[Tool]: List of discovered tool instances.
        """
        tools = []

        if str(self.tools_dir) not in sys.path:
            sys.path.insert(0, str(self.tools_dir))

        for root, _, files in os.walk(self.tools_dir):
            for file in files:
                if file.endswith('.py') and file not in self.excluded_files and not file.startswith('__'):
                    try:
                        rel_path = Path(root).relative_to(self.tools_dir)
                        if rel_path == Path('.'):
                            module_name = Path(file).stem
                        else:
                            module_name = f"{rel_path.as_posix().replace('/', '.')}.{Path(file).stem}"

                        module = importlib.import_module(module_name)

                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and
                                issubclass(obj, Tool) and
                                obj != Tool and
                                not name.startswith('_') and
                                name not in self.excluded_classes):
                                try:
                                    tools.append(obj())
                                except TypeError:
                                    continue
                    except ImportError as e:
                        print(f"Error importing {file}: {e}")
                        continue

        return tools