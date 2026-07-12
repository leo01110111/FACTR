---
name: factr-cluster-setup
description: Set up and run FACTR training on a SLURM cluster (e.g. CMU Babel) from a fresh checkout. Use when an agent is on a cluster node and the checkpoints, dataset buffer, visual features, or conda env are missing, or when asked to write an sbatch job for training. Explains why gitignored artifacts don't travel with git and how to obtain each one.
---

# Running FACTR on a SLURM cluster (Babel)

## The core gotcha

The repo's runtime artifacts are **gitignored** — `checkpoints/`,
`processed_data/`, `visual_features/`. They never arrive via `git pull`, and they
are **not** stored in wandb (the code logs metrics only, not checkpoint artifacts).
A fresh clone therefore has code but no data, no features, no checkpoints. They live
only on the machine where they were created (e.g. the workstation
`maxlab-host-001.ml.cmu.edu`, user `leo`, repo `/home/leo/FACTR`).

Do not conclude "there is no checkpoint" from a `find` on the cluster — check the
originating workstation, then transfer or regenerate.

## What each artifact needs

| Artifact | How to obtain on the cluster |
|---|---|
| conda env `factr` | recreate from `env.yaml` (see below) — not the hardcoded workstation path |
| `visual_features/vit_base/SOUP_1M_DH.pth` (427 MB) | **download**: `bash scripts/download_features.sh` (public URL) |
| `processed_data/<ds>/buf.pkl` (~78 MB) + `rollout_config.yaml` | **transfer** from workstation (not downloadable) or regenerate via `factr-process-data` |
| `checkpoints/<run>/` (~2.5 GB/ckpt) | transfer **only if resuming**; for new training you don't need it |

**Recommendation: start fresh** unless explicitly extending a finished run — a new
20k-step run only needs the 78 MB buffer + downloadable features, versus 2.5 GB per
checkpoint.

## Step 1 — conda env (inside a GPU allocation)

```bash
cd /path/to/FACTR
conda env create -f env.yaml   # env name: factr
conda activate factr           # scripts call bare `python`
```

Build it in an interactive GPU allocation so `pytorch-cuda` resolves the right CUDA.
The scripts use bare `python`; do not rely on any hardcoded `/home/leo/...` path.

## Step 2 — features (download)

```bash
bash scripts/download_features.sh
ls visual_features/vit_base/SOUP_1M_DH.pth
```

No internet on compute nodes? Download on a login node / via proxy, or transfer.

## Step 3 — dataset buffer (transfer, required)

`buf.pkl` and `rollout_config.yaml` must land in the **same** directory. Pull both:

```bash
mkdir -p processed_data/box-in-box
rsync -avP leo@maxlab-host-001.ml.cmu.edu:/home/leo/FACTR/processed_data/box-in-box/ \
      processed_data/box-in-box/
md5sum processed_data/box-in-box/buf.pkl   # box-in-box: c91fea15d2ee4f6fdeea452b4ca59ed5
```

If the cluster can't SSH out, push from the workstation instead. If transfer is
impossible, regenerate with the `factr-process-data` workflow from the raw data.

## Step 4 — sbatch wrapper (replace nohup)

```bash
#!/bin/bash
#SBATCH --job-name=factr-box-in-box
#SBATCH --partition=general          # adjust
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12           # >= num_workers (10) + headroom
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue                    # survive preemption
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source ~/.bashrc
conda activate factr
cd $SLURM_SUBMIT_DIR
export WANDB_MODE=online
bash scripts/train_bc.sh
```

Before submitting: set your own `wandb_entity` in `scripts/train_bc.sh`, ensure
`wandb login` / `~/.netrc` is set up on the node (else it stalls on a prompt), and
add `exp_name=<name>` so it doesn't default to `test`. On `--requeue`, the job
re-runs from step 0 unless `exp_name` points at a persistent dir set up for resume.

Submit and verify:

```bash
sbatch scripts/train_babel.sbatch
grep -a "wandb.ai\|Starting at Global Step\|Traceback" slurm-<jobid>.out
```

Full detail (including "resume the 40k run" path): `docs/box-in-box-cluster-setup.md`.
Training mechanics: the `train-factr-policy` skill.
