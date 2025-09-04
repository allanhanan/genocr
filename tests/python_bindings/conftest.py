import pytest
import sys
from pathlib import Path

# Add the python_bindings directory to Python path so we can import genocr
# From tests/python_bindings/, go up 2 levels to project root, then into python_bindings
python_bindings_dir = Path(__file__).parent.parent.parent / "python_bindings"
sys.path.insert(0, str(python_bindings_dir))
