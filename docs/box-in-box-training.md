# Training on the box-in-box dataset

Instructions for training the FACTR BC policy on the box-in-box dataset. Assumes
a cold start — everything you need is below.

## Prerequisites (verify before running)

- **Python**: use the conda env, not the system python (there is no bare `python`):
  `/home/leo/miniconda3/envs/factr/bin/python`
- **Dataset buffer**: `processed_data/box-in-box/buf.pkl` (~78 MB) must exist.
- **Pretrained visual features**: `visual_features/vit_base/SOUP_1M_DH.pth` (~427 MB) must exist.
- **wandb**: online logging works out of the box — credentials for `api.wandb.ai`
  are in `~/.netrc` (logged in as `leokswang`). The scripts export
  `WANDB_MODE=online`. Runs land in project `factr`, entity
  `leokswang-carnegie-mellon-university`.
- **GPU**: single GPU (`CUDA_VISIBLE_DEVICES=0`). `batch_size=32` fits a 32 GB
  RTX 5090; 128 OOMs. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set.

Quick check:

```bash
cd /home/leo/FACTR
ls -la processed_data/box-in-box/buf.pkl visual_features/vit_base/SOUP_1M_DH.pth
```

## Fresh training run

Trains from scratch for 20000 iterations with a linear blur curriculum
(pixel-space blur, scale 5 → 0). Config lives in `scripts/train_bc.sh`.

```bash
cd /home/leo/FACTR
nohup bash scripts/train_bc.sh > train.out 2>&1 &
```

Key settings in that script: `task=ur_left`, `curriculum.scheduler=linear`,
`start_scale=5`, `stop_scale=0`. The run name/checkpoint dir come from `exp_name`
(defaults to `test` in `factr/cfg/train_bc.yaml`) — set `exp_name=<name>` to give
it a clean name and its own dir under `checkpoints/`.

## Resuming / extending a finished run

The completed 20000-step run lives in `checkpoints/test/` (curriculum ended at
scale 0 / sharp). To continue it to 40000 steps **without resetting the
curriculum**, use `scripts/train_bc_resume.sh`:

```bash
cd /home/leo/FACTR
nohup bash scripts/train_bc_resume.sh > train_resume.out 2>&1 &
```

Why the curriculum is not reset: the resume script passes
`curriculum.scheduler=no`, and `get_scale()` (in `factr/utils.py`) returns `0`
for `scheduler=no` — i.e. full-resolution/sharp images, matching where the linear
curriculum ended. It does **not** restart blur at scale 5.

### How resume works (important)

`misc.init_job()` decides fresh-vs-resume by looking for `exp_config.yaml` in the
hydra run dir `checkpoints/${exp_name}`:

- **If `exp_config.yaml` exists** → resumes, loading `rollout/latest_ckpt.ckpt`
  from that dir and starting at its saved `global_step`.
- **If not** → fresh run starting at step 0.

To bootstrap a resume from an existing run, copy that run's dir so the resume
target has both `exp_config.yaml` and `rollout/latest_ckpt.ckpt`, e.g.:

```bash
cp -r checkpoints/test checkpoints/test_resume
```

The **runtime** config (max_iterations, scheduler, etc.) comes from the hydra
overrides in the script, not from the copied `exp_config.yaml` (that stored copy
is only used to detect resume + supply the wandb config/id). So set
`max_iterations=40000`, `curriculum.scheduler=no` on the command line, as the
resume script already does.

## Verifying the run is healthy

Give it ~1–2 min to build buffers, restore the checkpoint, and init wandb, then:

```bash
grep -a "wandb.ai\|Starting at Global Step\|Traceback" train_resume.out | head
```

You should see a wandb run URL, `Starting at Global Step <N>`, and no traceback.
Eval runs every 200 steps (`eval_freq`) and logs `eval/task_loss`,
`eval/action_l2`, `eval/action_lsig` to wandb.

## Gotchas

- **Do not** log offline — keep `WANDB_MODE=online` (the scripts set it).
- `factr/task.py` `BCTask.eval` previously crashed with
  `NameError: average_weights` (incomplete attention-weight logging). This is
  fixed; if you reintroduce attention logging, compute `average_weights` before
  logging it.
- Log files (`*.out`) are gitignored — do not commit them.
- Checkpoints keep only the 2 most recent `ckpt_*.ckpt` plus
  `rollout/latest_ckpt.ckpt` (see `save_checkpoint`, `save_freq=2000`).
