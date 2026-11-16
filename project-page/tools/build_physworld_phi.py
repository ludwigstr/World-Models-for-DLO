import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
d = json.loads((root / "results/physworld/physworld_shared_phi.json").read_text(encoding="utf-8"))


def sig(v, n=4):
    return float(f"{v:.{n}g}")


def chan(key, c):
    return [[sig(math.exp(node[c])) for node in cable] for cable in d[key]]


shared = {
    "clip": "all 36 clips",
    "mean": round(d["shared_fit"]["mean"], 3),
    "before": round(d["baselines"]["mean twin Phi"]["mean"], 3),
    "frames": None,
    "E": {"fit": chan("phi_shared_raw", 0), "twin": chan("phi_twin_mean", 0)},
    "rho": {"fit": chan("phi_shared_raw", 1), "twin": chan("phi_twin_mean", 1)},
}
out = {"source": "results/physworld/physworld_shared_phi.json", "box": 1.5, "shared": True, "clips": [shared]}
dst = root / "project-page" / "static" / "js" / "physworld_phi.js"
dst.write_text("window.PHYSWORLD_PHI = " + json.dumps(out, separators=(",", ":")) + ";\n", encoding="utf-8")
print("shared Phi", shared["mean"], "before", shared["before"], "->", dst)
