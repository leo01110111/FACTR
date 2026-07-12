---
name: train-factr-policy
description: Train or resume the FACTR behavior-cloning policy on a robobuf dataset (e.g. box-in-box). Use when asked to train/resume/continue a policy, launch a training run, or set up wandb logging for FACTR. Covers the curriculum, resume mechanism, wandb online logging, and the common failure modes.
---

# Training the FACTR BC policy

## Prerequisites (check these first — most failures are missing artifacts)

These paths are **gitignored** and exist only where they were created; they never
travel with `git pull`. Verify before launching:

- **Conda env** `factr` — activate it (`conda activate factr`); scripts call bare
  `python`. If missing: `conda env create -f env.yaml`.
- **Dataset buffer** `processed_data/<dataset>/buf.pkl` **and** `rollout_config.yaml`
  in the *same* directory — `train_bc_policy.py` reads `rollout_config.yaml` from
  the buffer's parent dir. Generate with the `factr-process-data` workflow if absent.
- **Visual features** `visual_features/vit_base/SOUP_1M_DH.pth` — get via
  `bash scripts/download_features.sh` (public download, no transfer needed).
- **wandb** — must be logged in (`~/.netrc` with `api.wandb.ai`, or `wandb login`),
  else `wandb.init` stalls on a prompt. Set `wandb_entity` in the script to your own.

There is no bare `python` on some machines — inside the activated env it works;
otherwise find the env's python (`conda run -n factr which python`).

## Fresh run

`scripts/train_bc.sh` trains from scratch: 20000 iters, `task=ur_left`, linear blur
curriculum (pixel-space, scale 5→0), `batch_size=32`. Edit `buffer_path`,
`wandb_entity`, and add `exp_name=<clean_name>` (default is `test`). Then:

```bash
conda activate factr
WANDB_MODE=online nohup bash scripts/train_bc.sh > train.out 2>&1 &
```

Checkpoints and logs go to `checkpoints/${exp_name}/` (hydra run dir).

## Resume / extend a finished run

`scripts/train_bc_resume.sh` continues a run to 40000 iters with
`curriculum.scheduler=no`. Launch with `nohup bash scripts/train_bc_resume.sh`.

**How resume is decided** (`misc.init_job`): it looks for `exp_config.yaml` in the
hydra run dir `checkpoints/${exp_name}`. If present → resumes from
`rollout/latest_ckpt.ckpt` at its saved `global_step`. If absent → fresh from step 0.
To bootstrap a resume from an existing run, `cp -r checkpoints/<old> checkpoints/<new>`
so the target has both `exp_config.yaml` and `rollout/latest_ckpt.ckpt`.

**Runtime config comes from the CLI overrides in the script, not the copied
`exp_config.yaml`** (that copy only detects resume + supplies the wandb config/id).

## Curriculum must not be reset when resuming — how it works

`get_scale()` in `factr/utils.py` maps step→blur scale. With `scheduler=no` it
returns **0** (full-resolution/sharp), matching where a linear 5→0 curriculum ended.
So resuming a completed run with `scheduler=no` keeps images sharp — it does **not**
restart blur at scale 5. Never resume a finished run with `scheduler=linear` unless
you intend to re-run the curriculum. Note `get_scale` asserts `cur_step <= max_step`,
so `max_iterations` must be ≥ the resume step.

## Verify a run is healthy

Give it 1–2 min (builds buffers, restores ckpt, inits wandb), then:

```bash
grep -a "wandb.ai\|Starting at Global Step\|Traceback" train.out | head
```

Want: a wandb run URL, `Starting at Global Step <N>`, no traceback. Eval runs every
`eval_freq=200` steps, logging `eval/task_loss`, `eval/action_l2`, `eval/action_lsig`.

## Known failure modes

- **`NameError: average_weights` in `factr/task.py` `BCTask.eval`** — incomplete
  attention-weight logging (already fixed by removing the dead log line). If you
  re-add attention logging, compute `average_weights` before logging it.
- **Offline wandb** — keep `WANDB_MODE=online`; the scripts set it.
- **OOM** — `batch_size=128` OOMs a 32 GB GPU; use 32 (fits), maybe 64.
- **Missing `rollout_config.yaml`** — trainer crashes building the rollout dir if
  it isn't next to `buf.pkl`.

For running on a SLURM cluster, see the `factr-cluster-setup` skill and
`docs/box-in-box-cluster-setup.md`. Full walkthrough: `docs/box-in-box-training.md`.
