import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = Path("extern/DLO-Lab")
PATCHED = ROOT.parent / "extern/DLO-Lab-perjoint"

DLO_LAB = Path(os.environ.get("WIRE_DLOLAB", PATCHED))
