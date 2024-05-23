from policies.learner import Learner
import numpy as np
from utils import logger
import torch
from torchkit import pytorch_utils as ptu
from functools import partial

class LearnerRegular(Learner):
    def __init__(self, *args, **kwargs):
        super(LearnerRegular, self).__init__(*args, **kwargs)
        self.lens = [self.eval_env.eval_length + i for i in [- int(self.eval_env.eval_length / 2), 0, 1, 2, 3, 4, 8, 16, 32]]
    
    @torch.no_grad()
    def collect_rollouts(self, num_rollouts, random_actions=False):
        obs_list, act_list, rew_list, term_list, next_obs_list = self.train_env.rollout(num_rollouts)
        if np.random.rand() < self.agent.algo.epsilon_schedule(self._n_env_steps_total) and not random_actions:
            action, reward, internal_state = self.agent.get_initial_info()
            current_actions = ptu.from_numpy(act_list).unsqueeze(-1).long()
            
            rewards = ptu.from_numpy(rew_list)
            observs = ptu.from_numpy(obs_list)
            current_actions = ptu.id_to_onehot(current_actions, self.agent.action_dim)
            prev_actions = ptu.zeros_like(current_actions)
            prev_actions[:1] = action
            prev_actions[1:] = current_actions[:-1]
            rewards[1:] = rewards[:-1].clone()
            rewards[:1] = reward
            prev_actions = prev_actions.unsqueeze(1)
            rewards = rewards.unsqueeze(1).unsqueeze(-1)
            observs = observs.unsqueeze(1)
            current_actions = current_actions.unsqueeze(1)
            q = self.agent.critic(prev_actions, rewards, observs, current_actions)
            act_list[-1] = q[-1, 0].argmax().item()
            # act_list = ptu.id_to_onehot(act_list[None], self.agent.action_dim)
            rew_list[-1] = (self.train_env.action_mapping[act_list[-1]] == self.train_env.target) + self.train_env.penalty * (self.train_env.action_mapping[act_list[-1]] == '*')
            

        self.policy_storage.add_episode(
            observations=obs_list,  # (L, dim)
            actions=act_list[..., None],  # (L, dim)
            rewards=rew_list[..., None],  # (L, dim)
            terminals=term_list[..., None],  # (L, 1)
            next_observations=next_obs_list,  # (L, dim)
        )
        
        before_env_steps = self._n_env_steps_total
        self._n_env_steps_total += obs_list.shape[0]
        self._n_rollouts_total += 1
        return self._n_env_steps_total - before_env_steps

    @torch.no_grad()
    def evaluate(self, deterministic=True):
        self.agent.eval()
        returns_per_episode = np.zeros(self.config_env.eval_episodes)
        success_rate = np.zeros(self.config_env.eval_episodes)
        total_steps = np.zeros(self.config_env.eval_episodes)
        trajs = []
        
        obs_list, act_list, rew_list, term_list, next_obs_list, targets = self.eval_env.rollout(self.config_env.eval_episodes)
        action, reward, internal_state = self.agent.get_initial_info()
        action = action.repeat(self.config_env.eval_episodes, 1)
        reward = reward.repeat(self.config_env.eval_episodes, 1)
        current_actions = ptu.from_numpy(act_list).long()
        
        rewards = ptu.from_numpy(rew_list).unsqueeze(-1)
        observs = ptu.from_numpy(obs_list)
        current_actions = torch.nn.functional.one_hot(current_actions, self.agent.action_dim)
        prev_actions = ptu.zeros_like(current_actions)
        prev_actions[0] = action
        prev_actions[1:] = current_actions[:-1]
        rewards[1:] = rewards[:-1].clone()
        rewards[0] = reward
        q = self.agent.critic(prev_actions, rewards, observs, current_actions)
        final_actions = q[-1].argmax(dim=-1).cpu().numpy()
        for i in range(self.config_env.eval_episodes):
            is_success = float(self.eval_env.action_mapping[final_actions[i]] ==  targets[i])
            is_penalty = (self.eval_env.action_mapping[final_actions[i]] == '*')
            returns_per_episode[i] = is_success + self.eval_env.penalty * is_penalty
            success_rate[i] = is_success
            total_steps[i] = act_list.shape[0]
        self.agent.train()
        return returns_per_episode, success_rate, total_steps, trajs

    def evaluate_on_len(self, deterministic=True):
        returns_per_episodes = []
        success_rates = []
        for len in self.lens:
            self.eval_env.set_eval_len(len)
            returns_per_episode, success_rate, total_steps, trajs = self.evaluate(deterministic=deterministic)
            returns_per_episodes.append(np.mean(returns_per_episode))
            success_rates.append(np.mean(success_rate))
            print(f"Completed length {len}. Average return: {np.mean(returns_per_episode)}, success rate: {np.mean(success_rate)}")
            
        return returns_per_episodes, success_rates
    
    def log_on_len(self):
        returns_per_episodes, success_rates = self.evaluate_on_len(deterministic=True)
        for len, ret, success in zip(self.lens, returns_per_episodes, success_rates):
            logger.record_step("length", len)
            logger.record_tabular("return", ret)
            logger.record_tabular("success", success)
            logger.dump_tabular()