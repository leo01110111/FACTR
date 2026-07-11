# ---------------------------------------------------------------------------
# Offline evaluation for a trained BC policy: action-prediction error on the
# held-out (test) split, with no robot required. Mirrors the training setup so
# the numbers are directly comparable to what the policy saw during training.
# ---------------------------------------------------------------------------

import os
import yaml
import hydra
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd

import factr.misc  # noqa: F401  registers OmegaConf resolvers (transform/len/add/...)


DIM_LABELS = [
    "ur_j0", "ur_j1", "ur_j2", "ur_j3", "ur_j4", "ur_j5", "gripper",
]


@hydra.main(version_base=None, config_path="cfg", config_name="train_bc.yaml")
def main(cfg):
    ckpt_dir = Path(get_original_cwd()) / "checkpoints" / cfg.exp_name / "rollout"
    ckpt_path = ckpt_dir / "latest_ckpt.ckpt"
    rollout_config = yaml.safe_load((ckpt_dir / "rollout_config.yaml").read_text())
    ac_norm = rollout_config["norm_stats"]["action"]
    ac_std = np.array(ac_norm["std"])
    labels = DIM_LABELS if len(DIM_LABELS) == len(ac_std) else \
        [f"a{i}" for i in range(len(ac_std))]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = hydra.utils.instantiate(cfg.agent)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    agent.load_state_dict(state["model"])
    agent = agent.eval().to(device)
    print(f"Loaded {ckpt_path} (train step {state['global_step']})")

    task = hydra.utils.instantiate(
        cfg.task, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )

    # Accumulate per-dimension L1 over the first (current) action in each chunk,
    # weighted by the loss mask, in NORMALIZED units. Also track sign agreement.
    abs_err_sum = np.zeros(len(ac_std))
    weight_sum = np.zeros(len(ac_std))
    sign_agree_sum = 0.0
    sign_count = 0.0
    total_l1 = []

    for (imgs, obs), actions, mask in task.test_loader:
        imgs = {k: v.to(device) for k, v in imgs.items()}
        obs, actions, mask = obs.to(device), actions.to(device), mask.to(device)
        with torch.no_grad():
            pred = agent.get_actions(imgs, obs)
            total_l1.append(agent(imgs, obs, actions.reshape(actions.shape[0], -1),
                                  mask.reshape(mask.shape[0], -1)).item())

        # actions/pred: (B, ac_chunk, ac_dim); mask same shape
        err = (torch.abs(pred - actions) * mask).sum((0, 1)).cpu().numpy()
        w = mask.sum((0, 1)).cpu().numpy()
        abs_err_sum += err
        weight_sum += w

        agree = ((torch.sign(pred) == torch.sign(actions)).float() * mask)
        sign_agree_sum += agree.sum().item()
        sign_count += mask.sum().item()

    per_dim_norm = abs_err_sum / np.maximum(weight_sum, 1e-8)
    per_dim_real = per_dim_norm * ac_std  # undo z-score -> physical units

    print(f"\nHeld-out test set  |  mean L1 loss (normalized): {np.mean(total_l1):.4f}")
    print(f"Sign agreement: {100 * sign_agree_sum / max(sign_count, 1e-8):.1f}%\n")
    print(f"{'dim':<9}{'L1 (norm)':>12}{'L1 (real)':>14}   units")
    print("-" * 52)
    for i, lab in enumerate(labels):
        units = "rad" if lab.startswith("ur_") else "0-255"
        print(f"{lab:<9}{per_dim_norm[i]:>12.4f}{per_dim_real[i]:>14.4f}   {units}")


if __name__ == "__main__":
    main()
