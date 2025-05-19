# This file is intentionally empty.
# Path configuration is handled by pyproject.toml

import os
import sys

# Add the src directory to the Python path for pytest to find the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))