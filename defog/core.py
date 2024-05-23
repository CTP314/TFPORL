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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    done_flags = np.zeros(episodes, dtype=np.bool8)

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


def train(cfg, seed, log_dict, idx, logger, barrier, buffer_dir):
    using_mp = barrier is not None
    utils.config_logging("main_mp.log" if using_mp else "main.log")
    env_name = cfg.env.env_name
    eval_env = gym.vector.make(env_name + '-v4', render_mode="rgb_array", num_envs=cfg.train.eval_episodes, asynchronous=False, wrappers=RecordEpisodeStatistics)
    utils.set_seed_everywhere(eval_env, seed)

    state_dim = utils.get_space_shape(eval_env.observation_space, is_vector_env=True)
    action_dim = utils.get_space_shape(eval_env.action_space, is_vector_env=True)
    drop_cfg = cfg.buffer.drop_cfg
    buffer = instantiate(cfg.buffer, root_dir=buffer_dir, drop_cfg=drop_cfg, seed=seed)
    model = instantiate(cfg.model, state_dim=state_dim, action_dim=action_dim, action_space=eval_env.envs[0].action_space, state_mean=buffer.state_mean, state_std=buffer.state_std, device=device)
    cfg = DotMap(OmegaConf.to_container(cfg.train, resolve=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step+1)/cfg.warmup_steps, 1))

    logger.info(f"Training seed {seed} for {cfg.train_steps} timesteps with {env_name} {buffer.dataset.title()} dataset")

    
    if using_mp:
        local_log_dict = {key: [] for key in log_dict.keys()}
    else:
        local_log_dict = log_dict
        for key in local_log_dict.keys():
            local_log_dict[key].append([])

    best_reward = -np.inf
    utils.write_to_dict(local_log_dict, 'rtg_target', cfg.rtg_target, using_mp)
    for timestep in range(1, cfg.train_steps + cfg.finetune_steps + 1):
        states, actions, rewards_to_go, timesteps, dropsteps, mask = buffer.sample(cfg.batch_size)
        # no need for attention mask for the model as we always pad on the right side, whose attention is ignored by the casual mask anyway
        state_preds, action_preds, return_preds = model.forward(states, actions, rewards_to_go, timesteps, dropsteps)
        action_preds = action_preds[mask]
        action_loss = F.mse_loss(action_preds, actions[mask].detach(), reduction='mean')
        utils.write_to_dict(local_log_dict, 'action_loss', action_loss.item(), using_mp)

        optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        scheduler.step()

        if timestep % cfg.eval_interval == 0:
            eval_mean, eval_std = eval(eval_env, model, cfg.rtg_target)
            utils.write_to_dict(local_log_dict, 'eval_steps', timestep - 1, using_mp)
            utils.write_to_dict(local_log_dict, 'eval_returns', eval_mean, using_mp)
            d4rl_score = utils.get_d4rl_normalized_score(env_name, eval_mean)
            utils.write_to_dict(local_log_dict, 'd4rl_score', d4rl_score, using_mp)
            logger.info(f"Seed: {seed}, Step: {timestep}, Eval mean: {eval_mean:.2f}, Eval std: {eval_std:.2f}")

            if eval_mean > best_reward:
                best_reward = eval_mean
                model.save(f'best_train_seed_{seed}' if timestep <= cfg.train_steps else f'best_finetune_seed_{seed}')
                logger.info(f'Seed: {seed}, Save best model at eval mean {best_reward:.4f} and step {timestep}')

        if timestep % cfg.plot_interval == 0:
            utils.sync_and_visualize(log_dict, local_log_dict, barrier, idx, timestep, f'{env_name} {buffer.dataset.title()}', using_mp)

        if timestep == cfg.train_steps:
            model.save(f'final_train_seed_{seed}')
            model.load(f'best_train_seed_{seed}')

            perf_drop_curve = get_perf_drop_curve(eval_env, model, cfg.rtg_target, cfg.eval_drop_ps, seed)
            for drop_perf in perf_drop_curve:
                utils.write_to_dict(local_log_dict, 'perf_drop_train', drop_perf, using_mp)
            if cfg.finetune_steps > 0 and model.drop_aware:
                logger.info(f"Finetuning seed {seed} for {cfg.finetune_steps} timesteps with {env_name} {buffer.dataset.title()} dataset")
                model.freeze_trunk()
                buffer.drop_fn.drop_p = drop_cfg.finetune_drop_p
                buffer.drop_fn.update_dropmask()
                best_reward = -np.inf # ensure we will save best finetune model at least once
    
    if cfg.finetune_steps > 0 and model.drop_aware:
        model.save(f'final_finetune_seed_{seed}')
        model.load(f'best_finetune_seed_{seed}')

        perf_drop_curve = get_perf_drop_curve(eval_env, model, cfg.rtg_target, cfg.eval_drop_ps, seed)
        for drop_perf in perf_drop_curve:
            utils.write_to_dict(local_log_dict, 'perf_drop_finetune', drop_perf, using_mp)

    utils.sync_and_visualize(log_dict, local_log_dict, barrier, idx, timestep, f'{env_name} {buffer.dataset.title()}', using_mp)
    logger.info(f"Finish training seed {seed} with everage eval mean: {eval_mean}")
    return eval_mean