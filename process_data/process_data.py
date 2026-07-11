# ---------------------------------------------------------------------------
# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning
# https://arxiv.org/abs/2502.17432
# Copyright (c) 2025 Jason Jingzhou Liu and Yulong Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------


import yaml
import hydra
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig

from utils_data_process import sync_data_slowest, process_image, gaussian_norm, generate_robobuf

@hydra.main(version_base=None, config_path="cfg", config_name="default")
def main(cfg: DictConfig):

    input_path = cfg.input_path
    output_path = cfg.output_path

    rgb_obs_topics = list(cfg.cameras_topics)
    state_obs_topics = list(cfg.obs_topics)

    action_source_topics = list(cfg.action_source_topics)
    action_publish_topics = list(cfg.action_publish_topics)
    action_source_dims = list(cfg.action_source_dims) if cfg.get("action_source_dims") else \
        [None] * len(action_source_topics)

    assert len(state_obs_topics) > 0, "Require low-dim observation topics"
    assert len(rgb_obs_topics) > 0, "Require visual observation topics"
    assert len(action_source_topics) > 0, "Require action source (observation) topics"
    assert len(action_source_topics) == len(action_publish_topics), \
        "action_source_topics and action_publish_topics must correspond one-to-one"
    assert len(action_source_dims) == len(action_source_topics), \
        "action_source_dims must have one entry per action source topic"


    data_folder = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    all_topics = list(dict.fromkeys(state_obs_topics + rgb_obs_topics + action_source_topics))

    all_episodes = sorted([f for f in data_folder.iterdir() if f.name.startswith('ep_') and f.name.endswith('.pkl')])

    trajectories = []
    all_states = []
    all_actions = []
    action_dims = None
    pbar = tqdm(all_episodes)
    for episode_pkl in pbar:
        with open(episode_pkl, 'rb') as f:
            traj_data = pickle.load(f)
        traj_data, avg_freq = sync_data_slowest(traj_data, all_topics)
        pbar.set_postfix({'avg_freq': f'{avg_freq:.1f} Hz'})

        traj = {}
        # action[t] = next-step observation of the source topics, so the final
        # observation has no action and is dropped: num_steps = len - 1.
        num_steps = len(traj_data[action_source_topics[0]]) - 1
        assert num_steps > 0, f"Episode {episode_pkl.name} too short to form actions"
        traj['num_steps'] = num_steps
        traj['states'] = np.concatenate(
            [np.array(traj_data[topic])[:num_steps] for topic in state_obs_topics], axis=-1
        )
        action_list = []
        for topic, keep_dims in zip(action_source_topics, action_source_dims):
            # shift forward by one: the action at step t is the observation at t+1
            actions = np.array(traj_data[topic])[1:num_steps + 1]
            if keep_dims is not None:
                # keep only the leading commandable columns (e.g. gripper position)
                actions = actions[:, :keep_dims]
            action_list.append(actions)
        traj['actions'] = np.concatenate(action_list, axis=-1)
        # record each source topic's dimension so rollout can split the flat action
        if action_dims is None:
            action_dims = [arr.shape[-1] for arr in action_list]

        all_states.append(traj['states'])
        all_actions.append(traj["actions"])

        for cam_ind, topic in enumerate(rgb_obs_topics):
            enc_images = traj_data[topic][:num_steps]
            processed_images = [process_image(img_enc) for img_enc in enc_images]
            traj[f'enc_cam_{cam_ind}'] = processed_images
        trajectories.append(traj)

    # map each publish (command) topic to its inferred action dimension
    action_config = {topic: int(dim) for topic, dim in zip(action_publish_topics, action_dims)}
        
    # normalize states and actions
    state_norm_stats = gaussian_norm(all_states)
    action_norm_stats = gaussian_norm(all_actions)
    norm_stats = dict(state=state_norm_stats, action=action_norm_stats)
    
    # dump data buffer
    buffer_name = "buf"
    buffer = generate_robobuf(trajectories)
    with open(output_dir / f"{buffer_name}.pkl", "wb") as f:
        pickle.dump(buffer.to_traj_list(), f)
    
    # dump rollout config
    obs_config = {
        'state_topics': state_obs_topics,
        'camera_topics': rgb_obs_topics,
    }
    rollout_config = {
        'obs_config': obs_config,
        'action_config': action_config,
        'norm_stats': norm_stats
    }
    with open(output_dir / "rollout_config.yaml", "w") as f:
        yaml.dump(rollout_config, f, sort_keys=False)
        
if __name__ == "__main__":
    main()