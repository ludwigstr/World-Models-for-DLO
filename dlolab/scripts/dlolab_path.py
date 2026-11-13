import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path("extern/DLO-Lab")
PATCHED = ROOT / "extern/DLO-Lab-perjoint"

DLO_LAB = Path(os.environ.get("WIRE_DLOLAB", PATCHED))
