"""Pytest configuration — add project root to sys.path for top-level modules."""

import sys
from pathlib import Path

# Add project root so we can import analyze_conventions, main, etc.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
