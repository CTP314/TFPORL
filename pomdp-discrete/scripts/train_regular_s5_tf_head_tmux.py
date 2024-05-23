import os
import time

if __name__ == "__main__":
    # os.walk()
    for n_head, gpu_id in zip([1, 4, 8], [4, 5, 6]):
        session_name = f"train_regular_s5_tf_head_{n_head}"
        os.system(f"tmux kill-session -t {session_name}")
        os.system(f"tmux new-session -d -s {session_name}")
        
        act_env_cmd = "conda activate tfporl"
        os.system(f"tmux send-keys -t {session_name} '{act_env_cmd}' Enter")
        
        train_cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python main.py \
        --config_env configs/envs/regular_parity.py \
        --config_env.env_name 25 \
        --config_rl configs/rl/dqn_default.py \
        --train_episodes 40000 \
        --config_seq configs/seq_models/gpt_default.py \
        --config_seq.model.seq_model_config.n_head {n_head} \
        --config_seq.sampled_seq_len -1 \
        --config_seq.model.action_embedder.hidden_size=0 \
        --config_rl.config_critic.hidden_dims="()" \
        --seed {int(time.time())} \
        --save_dir ablation '
        os.system(f"tmux send-keys -t {session_name} '{train_cmd}' Enter")
        os.system(f"tmux send-keys -t {session_name} '{train_cmd}' Enter")
        os.system(f"tmux send-keys -t {session_name} '{train_cmd}' Enter")
        
    