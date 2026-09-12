import sys
from pathlib import Path

# Ensure local src/ is importable for all tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
