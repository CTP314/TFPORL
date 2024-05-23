import os

if __name__ == "__main__":
    # os.walk()
    for root, dirs, files in os.walk("logs"):
        for file in files:
            if file == "flags.pkl":
                task = root.split("/")[1]
                seq_model = root.split("/")[3]
                print(f"Evaluating {task} with {seq_model} model")
                os.system(f"python eval_regular.py \
                    --config_env configs/envs/{task}.py \
                    --config_rl configs/rl/dqn_default.py \
                    --config_seq configs/seq_models/{seq_model}_default.py \
                    --load {root}"
                )