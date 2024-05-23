import utils
import torch
import numpy as np
import gymnasium as gym
from dotmap import DotMap
from omegaconf import OmegaConf
from model import DecisionTransformer
from hydra.utils import instantiate
from buffer import SequenceBuffer
import torch.nn.functional as F
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from drop_fn import DropWrapper
import os

DROPS = [0.0, 0.1, 0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.85, 0.9]

def get_perf_drop_curve(env: gym.vector.Env, model, rtg_target, drop_ps:list, seed):
    return_means = []
    for drop_p in drop_ps:
        drop_env = DropWrapper(env, drop_p, seed)
        mean, _ = eval(drop_env, model, rtg_target)
        return_means.append(mean)
    return return_means

@torch.no_grad()
def eval(env: gym.vector.Env, model: DecisionTransformer, rtg_target):
    # parallel evaluation with vectorized environment
    model.eval()
    
    episodes = env.num_envs
    reward, returns = np.zeros(episodes), np.zeros(episodes)
    done_flags = np.zeros(episodes, dtype=np.bool_)

    state_dim = utils.get_space_shape(env.observation_space, is_vector_env=True)
    act_dim = utils.get_space_shape(env.action_space, is_vector_env=True)
    max_timestep = model.max_timestep
    context_len = model.context_len
    timesteps = torch.arange(max_timestep, device=device)
    dropsteps = torch.zeros(max_timestep, device=device, dtype=torch.long)
    state, _ = env.reset(seed=[np.random.randint(0, 10000) for _ in range(episodes)])
    
    states = torch.zeros((episodes, max_timestep, state_dim), dtype=torch.float32, device=device)
    actions = torch.zeros((episodes, max_timestep, act_dim), dtype=torch.float32, device=device)
    rewards_to_go = torch.zeros((episodes, max_timestep, 1), dtype=torch.float32, device=device)

    reward_to_go, timestep, dropstep = rtg_target, 0, 0

    while not done_flags.all():
        states[:, timestep] = torch.from_numpy(state).to(device)
        rewards_to_go[:, timestep] = reward_to_go - torch.from_numpy(returns).to(device).unsqueeze(-1)
        dropsteps[timestep] = dropstep
        obs_index = torch.arange(max(0, timestep-context_len+1), timestep+1)
        _, action_preds, _ = model.forward(states[:, obs_index],
                                        actions[:, obs_index],
                                        rewards_to_go[:, obs_index - dropsteps[obs_index].cpu()], # drop rewards
                                        timesteps[None, obs_index],
                                        dropsteps[None, obs_index])

        action = action_preds[:, -1].detach()
        actions[:, timestep] = action

        state, reward, dones, truncs, info = env.step(action.cpu().numpy())
        dropstep = dropsteps[timestep].item() + 1 if info.get('dropped', False) else 0
        returns += reward * ~done_flags
        done_flags = np.bitwise_or(np.bitwise_or(done_flags, dones), truncs)
        timestep += 1
        
    model.train() 

    return np.mean(returns), np.std(returns)

path = 'runs/2024-03-24/06-41-59_buffer.dataset=medium,buffer.drop_cfg.drop_p=0.5,buffer.drop_cfg.finetune_drop_p=0.8,env=hopper,model=lru/'
cfg = OmegaConf.load(os.path.join(path, '.hydra/config.yaml'))

import csv
import os
import numpy as np
from tqdm import tqdm

with open('results.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'env', 'dataset', 'return'])

for file in os.listdir('runs/2024-03-24'):
    path = os.path.join('runs/2024-03-24', file)
    cfg = OmegaConf.load(os.path.join(path, '.hydra/config.yaml'))
    device = 'cuda:0'
    results = np.zeros((len(cfg.seeds), len(DROPS)))
    model_name = cfg.model._target_
    print(model_name)
    for seed in tqdm(cfg.seeds):
        dataset_dir = os.path.join(os.path.expanduser("~"), ".d4rl")

        buffer_dir = dataset_dir

        env_name = cfg.env.env_name
        eval_env = gym.vector.make(env_name + '-v4', render_mode="rgb_array", num_envs=cfg.train.eval_episodes, asynchronous=False, wrappers=RecordEpisodeStatistics)
        utils.set_seed_everywhere(eval_env, seed)

        state_dim = utils.get_space_shape(eval_env.observation_space, is_vector_env=True)
        action_dim = utils.get_space_shape(eval_env.action_space, is_vector_env=True)
        drop_cfg = cfg.buffer.drop_cfg
        buffer = instantiate(cfg.buffer, root_dir=buffer_dir, drop_cfg=drop_cfg, seed=seed)
        model = instantiate(cfg.model, state_dim=state_dim, action_dim=action_dim, action_space=eval_env.envs[0].action_space, state_mean=buffer.state_mean, state_std=buffer.state_std, device=device)

        train_cfg = DotMap(OmegaConf.to_container(cfg.train, resolve=True))

        print(f"Training seed {seed} for {train_cfg.train_steps} timesteps with {env_name} {buffer.dataset.title()} dataset")
        print(f'rtg target: {train_cfg.rtg_target}')

        model_path = os.path.join(path, f'models/best_train_seed_{seed}.pt')
        model.load_state_dict(torch.load(model_path))
        perf_drop_curve = get_perf_drop_curve(eval_env, model, train_cfg.rtg_target, DROPS, seed)
        
        print(f'perf drop curve: {perf_drop_curve}')
        results[seed] = perf_drop_curve
        
    with open('results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([model_name, cfg.env.env_name, cfg.buffer.dataset, results.mean(axis=0).tolist()])