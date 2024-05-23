import os
import time

GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
id = 0

for seed in [4]:
    for env in ['ant', 'cheetah', 'hopper', 'walker']:
        for task in ['v', 'p']:
            session_name = f"{env}-{task}-lru-{seed}"
            os.system(f"tmux kill-session -t {session_name}")
            os.system(f"tmux new-session -d -s {session_name}")
            
            os.system(f"tmux send-keys -t {session_name} 'source activate tfporljax' Enter")

            gpu_id = GPUS[id % len(GPUS)]
            id += 1
            
            train_cmd = f"""CUDA_VISIBLE_DEVICES={gpu_id} python main.py \
                    --config_env configs/envs/pomdps/pybullet_{task}.py \
                    --config_env.env_name {env} \
                    --config_rl configs/rl/td3_default.py \
                    --config_seq configs/seq_models/lru_default.py \
                    --config_seq.sampled_seq_len 64 \
                    --train_episodes 1500 \
                    --seed {int(time.time())} \
            """
        
            os.system(f"tmux send-keys -t {session_name} '{train_cmd}' Enter")
            time.sleep(1)