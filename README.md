# World Models for Deformable Linear Objects

Ten approaches predict the shape of four cables of a wire harness while their ends are moved, all on the same
36 recorded clips. Project page: https://ludwigstr.github.io/World-Models-for-DLO/

## Layout

```
<approach>/          code of one approach, one run.sh, requirements.txt
data/tracks/         the 36 tracked clips, one .npz per clip
data/videos/         camera footage of the 36 clips
results/<approach>/  parameters or network and per-clip errors of the runs behind the table
results/videos/      overlay videos of every approach on every clip
project-page/        sources of the project page
```

Each clip file holds the tracked cables `c0` to `c3` as arrays of shape (frames, 16 nodes, 2) and the frame rate
`fps`. All scripts read these files; the raw footage is not processed here.

## Scripts

The same name means the same step in every folder. A folder only has the steps its approach needs.

| File | Step |
|---|---|
| `simulate.py` | the simulator of the approach |
| `fit.py` | fits the simulator parameters to the 36 clips |
| `dataset.py` | generates training data with the simulator |
| `model.py` | the network |
| `train.py` | trains the network |
| `adapt.py` | fits the physical property vector of the trained network to the 36 clips (AdaptiGraph, PhysWorld) |
| `estimate.py` | online parameter estimation and prediction per clip (Caporali et al.) |
| `evaluate.py` | error of the network or twin on the 36 clips |
| `validate.py` | recomputes the error of fitted parameters in fresh processes |
| `run.sh` | runs all steps of the approach in order |

`dlolab/dlolab_patched/` is the DLO-Lab simulator with per-joint stiffness and rest angles and a corrected
gradient for scenes with several rods.

## Running

```
cd mujoco
pip install -r requirements.txt
./run.sh
```

Every chain starts from scratch: fits begin at fixed default values, networks are trained from random
initialisation, and no intermediate result is shipped with the code. Outputs go to `out/` inside the approach
folder (`data/` for PhysWorld).

`ropedreamer/run.sh` generates its training data with the fitted MuJoCo model and therefore needs
`mujoco/run.sh` to have finished first.

Workers, GPUs and the number of generations are set through environment variables at the top of each `run.sh`,
for example `WORKERS=24 GENS=60 ./run.sh`. Genesis runs with `genesis-world` 1.3.3; pass its interpreter as
`PY=/path/to/python ./run.sh`.
