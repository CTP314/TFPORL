import os

if __name__ == "__main__":
    # os.walk()
    GPUS = [0, 2, 3, 4]
    
    for task in ['s5', 'parity']:
        for len in [25, 50]:
            session_name = f"train_regular_{task}_{len}_gtrxl"
            os.system(f"tmux kill-session -t {session_name}")
            os.system(f"tmux new-session -d -s {session_name}")
            
            act_env_cmd = "conda activate tfporl"
            os.system(f"tmux send-keys -t {session_name} '{act_env_cmd}' Enter")
            
            gpu_id = GPUS.pop(0)
            train_cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python main.py \
            --config_env configs/envs/regular_s5.py \
            --config_env.env_name 25 \
            --config_rl configs/rl/dqn_default.py \
            --train_episodes 20000 \
            --config_seq configs/seq_models/gtrxl_default.py \
            --config_seq.sampled_seq_len -1 \
            --config_seq.model.action_embedder.hidden_size=0 \
            --config_rl.config_critic.hidden_dims="()"'
            os.system(f"tmux send-keys -t {session_name} '{train_cmd}' Enter")
            os.system(f"tmux send-keys -t {session_name} '{train_cmd}' Enter")
            os.system(f"tmux send-keys -t {session_name} '{train_cmd}' Enter")
        
    