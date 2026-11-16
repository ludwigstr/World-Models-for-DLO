import json, re, sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
pv = root / "results" / "videos"
npz_dir = root / "data" / "tracks36"

APPROACHES = {
    "physworld_4.5": "physworld",
    "caporali_15.8": "caporali",
    "phystwin_22.0": "phystwin",
    "adaptigraph_29.4": "adaptigraph",
    "genesis_12.7": "genesis",
    "yang_16.0": "yang",
}

clips = sorted(p.stem for p in npz_dir.glob("*.npz"))
frames = {}
for c in clips:
    z = np.load(npz_dir / f"{c}.npz", allow_pickle=True)
    frames[c] = int(z["c0"].shape[0])

per_clip, meta, stills = {}, {}, {}
for folder, key in APPROACHES.items():
    j = json.loads((pv / folder / "results.json").read_text(encoding="utf-8"))
    meta[key] = {"mean": j["mean_reproduced"], "metric": j["metric"], "approach": j["approach"]}
    per_clip[key] = {c: round(v["reproduced"], 3) for c, v in j["clips"].items()}
    vals = np.array(list(per_clip[key].values()))
    meta[key]["folder"] = folder
    s = {}
    for png in (pv / folder).glob("*.png"):
        m = re.match(rf"{key}_(\d{{8}}_\d{{6}})_frame(\d+)_(worst|middle)\.png", png.name)
        if m:
            s.setdefault(m.group(1), {})[m.group(3)] = {"file": png.name, "frame": int(m.group(2))}
    stills[key] = s
    missing = set(clips) - set(per_clip[key])
    assert not missing, (key, missing)

g0 = json.loads((root / "newton/state/newton_fit_everything_pinned_gen0.json").read_text())
newton_gen0 = {c: round(v, 3) for c, v in g0["clips"].items()}

out = {
    "clips": clips,
    "frames": frames,
    "perClip": per_clip,
    "meta": meta,
    "stills": stills,
    "newtonGen0": {"mean": g0["mean"], "clips": newton_gen0},
}
dst = root / "project-page" / "static" / "js" / "data.js"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("// Generated from results/videos/*/results.json and data/tracks.\n"
               "// Do not edit by hand; rerun the build script.\n"
               "window.BENCH_DATA = " + json.dumps(out, indent=1) + ";\n", encoding="utf-8")
print("clips", len(clips), "frames total", sum(frames.values()))
for k, m in meta.items():
    print(k, round(m["mean"], 2), "stills", len(stills[k]))
print("newton gen0", round(g0["mean"], 3))

