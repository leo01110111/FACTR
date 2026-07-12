---
name: factr-process-data
description: Convert raw ROS trajectory pickles into the robobuf buf.pkl dataset (plus rollout_config.yaml) that FACTR training consumes. Use when asked to process/prepare a new dataset, create buf.pkl, set up a new task's data, or when a training run is missing its buffer. Covers the topic config, the state/action/camera layout, and the task-config coupling.
---

# Processing raw data into a FACTR robobuf buffer

Turns per-episode ROS pickles into `processed_data/<dataset>/buf.pkl` +
`rollout_config.yaml`, the exact pair `train_bc_policy.py` loads.

## Input expectations

- `process_data/cfg/default.yaml` `input_path` points at a directory of episode
  files named `ep_*.pkl`. Each is `{"data": {topic: [...]}, "timestamps": {topic: [...]}}`.
- Topics are grouped in the config:
  - `cameras_topics` — RGB image streams (become `enc_cam_0`, `enc_cam_1`, … in order).
  - `obs_topics` — low-dim state topics, concatenated into the state vector.
  - `action_source_topics` — observation topics read **one step ahead** to form
    actions (action[t] = obs[t+1]); paired one-to-one with `action_publish_topics`
    (the real command topics) for rollout.
  - `action_source_dims` — per source topic, how many leading columns to keep
    (`null` = all). E.g. a gripper obs is `[position, current]`; keep `1` to drop the
    non-commandable current.

## Run it

```bash
conda activate factr
python process_data/process_data.py                      # uses cfg/default.yaml
# or override the dataset without editing the file:
python process_data/process_data.py input_path=/abs/raw/<ds> output_path=processed_data/<ds>
```

Outputs into `output_path/`:
- `buf.pkl` — robobuf trajectory list (states, actions, encoded camera frames).
- `rollout_config.yaml` — `obs_config` (state/camera topics), `action_config`
  (each command topic → its action-dim), and `norm_stats` (per-dim gaussian mean/std
  for state and action, from `gaussian_norm`).

## Key mechanics (so the output matches training)

- **Action = next-step observation.** The last frame has no action and is dropped:
  `num_steps = len(source_topic) - 1`. Episodes with ≤1 step assert-fail.
- **Sync**: `sync_data_slowest` aligns all topics to the slowest topic's rate.
- **Order matters**: state vector = `obs_topics` concatenated in listed order;
  cameras indexed in `cameras_topics` order. These must match what the task config
  and the trained policy expect.

## Coupling to the task config (must stay consistent)

The dims implied by the data must match the task config under
`factr/cfg/task/` (e.g. `ur_left.yaml`): observation dim = summed widths of
`obs_topics`, action dim = summed kept widths of `action_source_topics`, and camera
count = `len(cameras_topics)` (camera indices used by the buffer builder). If you
change topics or `action_source_dims`, update the task config's `obs_dim`,
`ac_dim`, and camera indices, or training will shape-mismatch. `rollout_config.yaml`
must sit next to `buf.pkl` — the trainer copies it into the run's rollout dir.

After producing `buf.pkl`, train with the `train-factr-policy` skill (point
`buffer_path` at `processed_data/<ds>/buf.pkl`). Note both `buf.pkl` and the raw
data are gitignored — they don't travel with the repo.
