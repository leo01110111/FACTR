# Running box-in-box training on the Babel cluster

The original training guide (`docs/box-in-box-training.md`) was written from the
**workstation** where everything already existed. On a fresh cluster node none of
the runtime artifacts are present, because they are all gitignored
(`checkpoints/`, `processed_data/`, `visual_features/`). This doc covers getting
them onto Babel.

## Where the artifacts actually live

Nothing is in git or in wandb (the code only logs metrics, not checkpoint
artifacts). Everything below is on the workstation:

- **host**: `maxlab-host-001.ml.cmu.edu` (user `leo`), repo at `/home/leo/FACTR`
- **conda env**: `/home/leo/miniconda3/envs/factr` (workstation only — recreate on
  Babel from `env.yaml`, see below)

| Artifact | Workstation path | Size | How to get it on Babel |
|---|---|---|---|
| Dataset buffer | `processed_data/box-in-box/buf.pkl` | 78 MB | **transfer** (not downloadable) |
| Rollout config | `processed_data/box-in-box/rollout_config.yaml` | 1.4 KB | **transfer** (must sit next to buf.pkl) |
| Visual features | `visual_features/vit_base/SOUP_1M_DH.pth` | 427 MB | **download** (public URL) |
| 20k-step checkpoint | `checkpoints/test/` | ~2.5 GB/ckpt | transfer **only if resuming** |
| 40k-step checkpoint | `checkpoints/test_resume/` | ~2.5 GB/ckpt | transfer **only if resuming** |

Integrity checksums (MD5):
- `buf.pkl` = `c91fea15d2ee4f6fdeea452b4ca59ed5`
- `rollout_config.yaml` = `82be193427bf367fd19129635209b272`

## Recommendation: start fresh on Babel

There **is** a completed run on the workstation (`checkpoints/test` = 20k steps,
`checkpoints/test_resume` = 40k steps, fully trained). But each checkpoint is
~2.5 GB and only lets you *continue* an already-finished run. Unless you
specifically need to extend that run, **start fresh** — you only need the 78 MB
buffer + downloadable features, and a fresh 20k-step run is cheap. The rest of
this doc assumes fresh training; a "resume instead" note is at the end.

## Step 1 — create the conda env on Babel

The repo ships the spec. On a node with conda:

```bash
cd /path/to/FACTR
conda env create -f env.yaml   # creates env named "factr"
conda activate factr           # scripts call bare `python`, so just activate
```

`env.yaml` pins python 3.9, pytorch+cuda, tensorflow 2.12, hydra, wandb, timm,
robobuf, r3m, and installs the repo itself (`-e .`). Do this inside a GPU/interactive
allocation so `pytorch-cuda` resolves against the right CUDA.

> The scripts now call bare `python` (the previously hardcoded
> `/home/leo/miniconda3/...` path was removed), so an activated `factr` env is all
> that's needed.

## Step 2 — get the visual features (download, no transfer)

```bash
cd /path/to/FACTR
bash scripts/download_features.sh
# pulls https://www.cs.cmu.edu/~data4robotics/release/features.zip and unzips to visual_features/
ls visual_features/vit_base/SOUP_1M_DH.pth   # verify
```

If the node has no internet, download on a login node / with a proxy, or transfer
the file the same way as the buffer below.

## Step 3 — transfer the dataset buffer (required, 78 MB)

`buf.pkl` is generated from raw ROS data and is not downloadable. Pull it and its
`rollout_config.yaml` from the workstation. **Both must land in the same
directory** — `train_bc_policy.py` reads `rollout_config.yaml` from the buffer's
parent dir.

From a Babel node that can reach the workstation:

```bash
cd /path/to/FACTR
mkdir -p processed_data/box-in-box
rsync -avP leo@maxlab-host-001.ml.cmu.edu:/home/leo/FACTR/processed_data/box-in-box/ \
      processed_data/box-in-box/
# verify integrity
md5sum processed_data/box-in-box/buf.pkl   # expect c91fea15d2ee4f6fdeea452b4ca59ed5
```

If Babel can't SSH out to the workstation, push from the workstation instead
(`rsync -avP processed_data/box-in-box/ <you>@<babel-node>:/path/to/FACTR/processed_data/box-in-box/`).

If transfer is impossible, regenerate: copy the raw data from the workstation
(`/home/leo/factr_teleop_ur7e/raw_data/box-in-box`) and run
`python process_data/process_data.py` (config: `process_data/cfg/default.yaml`,
`input_path` points at that raw dir). Transferring the 78 MB buffer is far easier.

## Step 4 — sbatch wrapper

Replace `nohup` with SLURM. Example `scripts/train_babel.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=factr-box-in-box
#SBATCH --partition=general          # adjust to your partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12           # >= num_workers (10) + headroom
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue                    # survive preemption
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source ~/.bashrc
conda activate factr
cd $SLURM_SUBMIT_DIR                  # run from the repo root
export WANDB_MODE=online             # ensure online logging
bash scripts/train_bc.sh
```

Notes:
- `scripts/train_bc.sh` already sets `PYTORCH_CUDA_ALLOC_CONF`, `task=ur_left`,
  the linear blur curriculum (scale 5→0), `batch_size=32`, and the wandb entity.
  It runs 20000 iterations (`max_iterations` in `factr/cfg/train_bc.yaml`).
- Set `wandb_entity` in `scripts/train_bc.sh` to your own wandb entity if you are
  not `leokswang-carnegie-mellon-university`, and make sure `wandb login` /
  `~/.netrc` is set up on the node — otherwise it will try to prompt and stall.
- `--requeue` + preemption: on requeue the job re-runs `train_bc.sh` from step 0
  unless the checkpoint dir is set up for resume (see below). For a first run
  that's fine; for long jobs, point `exp_name` at a persistent dir and rely on the
  resume mechanism to pick up `rollout/latest_ckpt.ckpt`.
- Give the run a clean name: add `exp_name=box_in_box` to the `python` line in
  `scripts/train_bc.sh` (default is `test`).

Submit and verify:

```bash
sbatch scripts/train_babel.sbatch
# once it starts, tail the log:
grep -a "wandb.ai\|Starting at Global Step\|Traceback" slurm-<jobid>.out
```

You want a wandb run URL, `Starting at Global Step 0`, and no traceback. Eval runs
every 200 steps and logs `eval/task_loss`, `eval/action_l2`, `eval/action_lsig`.

## If you want to resume the 40k run instead of starting fresh

Transfer the checkpoint dir (2.5 GB) and use the resume path:

```bash
rsync -avP leo@maxlab-host-001.ml.cmu.edu:/home/leo/FACTR/checkpoints/test_resume/ \
      checkpoints/test_resume/
```

Then run `scripts/train_bc_resume.sh` (via sbatch). Resume works because
`checkpoints/test_resume/exp_config.yaml` exists → `misc.init_job` loads
`rollout/latest_ckpt.ckpt` and continues at its saved step. `scheduler=no` keeps
the curriculum at scale 0 (sharp), so it is **not** reset. Note the 40k run has
already hit `max_iterations=40000`; to train further, bump `max_iterations` in the
resume script above 40000.

## Recap of the two questions

- **Is there a checkpoint?** Yes, but only on the workstation
  (`maxlab-host-001.ml.cmu.edu:/home/leo/FACTR/checkpoints/{test,test_resume}`) —
  not in git, not in wandb, not on Babel. It's a *finished* run; for new training
  you don't need it.
- **Where is the factr conda env?** Only on the workstation
  (`/home/leo/miniconda3/envs/factr`). On Babel, create it from `env.yaml`
  (`conda env create -f env.yaml`). That's the source of the hardcoded-path
  breakage; the script now uses bare `python`.
