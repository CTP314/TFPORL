import os
import sys
import torch
import pickle
import numpy as np
from dotmap import DotMap
from drop_fn import get_drop_fn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

class SequenceBuffer():
    def __init__(self, env_name, dataset, context_len, root_dir, gamma, drop_cfg, sample_type='traj_length', seed=0) -> None:
        dataset_path = os.path.join(root_dir, 'datasets', f'{env_name.lower()}-{dataset}.pkl')
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        self.num_trajs = len(trajectories)
        self.dataset = dataset

        self.state_dim = trajectories[0]['observations'].shape[1]
        self.action_dim = trajectories[0]['actions'].shape[1]
        self.context_len = context_len
        self.size = sum([len(traj['observations']) for traj in trajectories]) + 1  # plus one for padding zeros

        self.states = np.zeros((self.size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.size, self.action_dim), dtype=np.float32)
        self.rewards_to_go = np.zeros((self.size,), dtype=np.float32)

        self.traj_length = np.zeros(self.num_trajs, dtype=np.int32)
        self.traj_sp = np.zeros(self.num_trajs, dtype=np.int32)  # trajectory start point
        self.traj_returns = np.zeros(self.num_trajs, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        traj_pointer = 0

        for i, traj in enumerate(trajectories):
            self.traj_sp[i] = traj_pointer
            observations, actions, rewards = traj['observations'], traj['actions'], traj['rewards']
            # observations, actions, rewards = traj['observations'], traj['actions'], -traj['rewards']
            # observations, actions, rewards = traj['observations'], traj['actions'], self.rng.uniform(-1, 1, size=traj['rewards'].shape)
            assert observations.shape[0] == actions.shape[0] == rewards.shape[0], 'observations, actions, rewards should have the same length'
            self.traj_length[i] = observations.shape[0]

            self.states[self.traj_sp[i]: self.traj_sp[i] + self.traj_length[i]] = observations
            self.actions[self.traj_sp[i]: self.traj_sp[i] + self.traj_length[i]] = actions
            self.rewards_to_go[self.traj_sp[i]: self.traj_sp[i] + self.traj_length[i]] = discount_cumsum(rewards, gamma)
            self.traj_returns[i] = np.sum(rewards)
            traj_pointer += self.traj_length[i]

        assert sample_type in ['uniform', 'traj_return', 'traj_length'], 'sample_type should be one of [uniform, traj_return, traj_length]'
        self.p_sample = np.ones(self.num_trajs) / self.num_trajs if sample_type == 'uniform' else self.traj_returns / \
            self.traj_returns.sum() if sample_type == 'traj_return' else self.traj_length / self.traj_length.sum()
        self.state_mean, self.state_std = self.states.mean(axis=0), self.states.std(axis=0)
        self.drop_fn = get_drop_fn(drop_cfg, self.size, self.traj_sp, self.rng)
            
    def sample(self, batch_size):
        selected_traj = self.rng.choice(np.arange(self.num_trajs), batch_size, replace=True, p=self.p_sample)
        selected_traj_sp = self.traj_sp[selected_traj]
        selected_offset = np.floor(self.rng.random(batch_size) * (self.traj_length[selected_traj] - self.context_len)).astype(np.int32).clip(min=0)
        selected_sp = selected_traj_sp + selected_offset
        selected_ep = selected_sp + self.traj_length[selected_traj].clip(max=self.context_len)

        # fill the index of those padded steps with -1, so that we can fetch the last step of the corresponding item, which is zero intentionally
        selected_index = selected_sp[:, None] + np.arange(self.context_len)
        selected_index = np.where(selected_index < selected_ep[:, None], selected_index, -1)
        masks = selected_index >= 0
        timesteps = selected_offset[:, None] + np.arange(self.context_len)  # we don't care about the timestep for those padded steps
        
        # update and get drop mask
        self.drop_fn.step()
        dropsteps = self.drop_fn.get_dropsteps(selected_index)
        observation_index = selected_index - dropsteps

        states = torch.as_tensor(self.states[observation_index, :]).to(dtype=torch.float32, device=device)
        actions = torch.as_tensor(self.actions[selected_index, :]).to(dtype=torch.float32, device=device)
        rewards_to_go = torch.as_tensor(self.rewards_to_go[observation_index, None]).to(dtype=torch.float32, device=device)
        timesteps = torch.as_tensor(timesteps).to(dtype=torch.int32, device=device)
        dropsteps = torch.as_tensor(dropsteps).to(dtype=torch.int32, device=device)

        return states, actions, rewards_to_go, timesteps, dropsteps, masks
