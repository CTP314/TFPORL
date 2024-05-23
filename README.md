<p align="center">
  <h1 align="center">Rethinking Transformers in Solving POMDPs</h1>
</p>

[![pytorch](https://img.shields.io/badge/Python-PyTorch-orange.svg)](https://www.pytorch.org)
[![JAX Code](https://img.shields.io/badge/JAX-Code-orange)](https://github.com/ikostrikov/jaxrl)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/CTP314/TFPORL/blob/master/LICENSE)


This repo is the official code release for the ICML 2024 conference paper:
<p align="center">&nbsp;<table><tr><td>
    <p align="center">
    <strong>
        <a href="https://github.com/CTP314/TFPORL">
Rethinking Transformers in Solving POMDPs
        </a><br/>
    </strong>
    Chenhao Lu<sup>1</sup>, Ruizhe Shi*<sup>1</sup>, Yuyao Liu*<sup>1</sup>, Kaizhe Hu<sup>2</sup>, Simon Shaolei Du<sup>2</sup>, Huazhe Xu<sup>13</sup><br>
    <b>The International Conference on Machine Learning (ICML) 2024</b><br>
    <sup>1</sup><em>Tsinghua Universtiy, IIIS&nbsp;&nbsp;</em>
    <sup>2</sup><em>University of Washington&nbsp;&nbsp;</em>
    <sup>3</sup><em>Shanghai Qi Zhi Institute&nbsp;&nbsp;</em><br>
    *Equal contribution. Order is decided by coin flip.
    </p>
</td></tr></table>&nbsp;</p>

# 🧾 Introduction


In this work, we challenge the suitability of Transformers
as sequence models in Partially Observable RL by leveraging regular language and circuit complexity theories. We advocate Linear RNNs as a promising alternative.


In the paper, we compare representative models including GPT, LSTM, and LRU on three different tasks to validate our theory through experiments. This codebase is used to reproduce the experimental results from the paper.

# 💻 Installation

## Regular Language Task & Pure Long-Term Memory Task

Run the following commands.

```
cd pomdp-discrete
conda create -n tfporl-discrete python=3.8
pip install -r requirements.txt
```

## Pybullet Occlusion Task

Run the following commands.

```
cd pomdp-discrete
conda create -n tfporl-continuous python=3.8
pip install -r requirements.txt
```

If you meet any problems, please refer to the guidance in [JAX](https://github.com/google/jax).

## Random Frame Dropping Task

### Environment
We can only guarantee the reproducibility with the environment configuration as below.
#### Install MuJoCo
First, you need to download the file from this [link](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) and `tar -xvf the_file_name` in the `~/.mujoco` folder. Then, run the following commands.
```bash
cd defog
conda create -n tfporl-defog python=3.8.17
```
After that, add the following lines to your `~/.bashrc` file:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/YOUR_PATH_TO_THIS/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```
Remember to `source ~/.bashrc` to make the changes take effect.

#### Install D4RL
Install D4RL by following the guidance in [D4RL](https://github.com/Farama-Foundation/D4RL).

Degrade the dm-control and mujoco package:
```bash
pip install mujoco==2.3.7
pip install dm-control==1.0.14
```

#### Install torch and other dependencies
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

### Dataset

To download original D4RL data, 
```bash
python download_d4rl_datasets.py
```

# 🛠️ Usage

## Regular Language Task & Pure Long-Term Memory Task

After installing packages, you can run the following script to reproduce results: 

```bash
cd pomdp-discrete
# for regular language tasks
python main.py \
        --config_env configs/envs/regular_parity.py \
        --config_env.env_name 25 \
        --config_rl configs/rl/dqn_default.py \
        --train_episodes 40000 \
        --config_seq configs/seq_models/gpt_default.py \
        --config_seq.model.seq_model_config.n_layer {n_layer} \
        --config_seq.sampled_seq_len -1 \
        --config_seq.model.action_embedder.hidden_size=0 \
        --config_rl.config_critic.hidden_dims="()"
# for Passive T-maze
python main.py \
    --config_env configs/envs/tmaze_passive.py \
    --config_env.env_name 50 \
    --config_rl configs/rl/dqn_default.py \
    --train_episodes 20000 \
    --config_seq configs/seq_models/lstm_default.py \
    --config_seq.sampled_seq_len -1 \
# for Passive Visual Match
python main.py \
    --config_env configs/envs/visual_match.py \
    --config_env.env_name 60 \
    --config_rl configs/rl/sacd_default.py \
    --shared_encoder --freeze_critic \
    --train_episodes 40000 \
    --config_seq configs/seq_models/gpt_cnn.py \
    --config_seq.sampled_seq_len -1 \
```

In the scripts, `env_name` is the max training length of regular langauge task. You can try other regular language tasks in `pomdp-discretes/configs/envs/`. and other sequence model in `pomdp-discretes/configs/seq_models/`.

Feel free to add other regular language in `pomdp-discretes/envs/regular.py` by input its DFA.

## Pybullet Occlusion Task

After installing packages, you can run the following script to reproduce results: 

```bash
python main.py \
    --config_env configs/envs/pomdps/pybullet_p.py \
    --config_env.env_name cheetah \
    --config_rl configs/rl/td3_default.py \
    --config_seq configs/seq_models/lstm_default.py \
    --config_seq.sampled_seq_len 64 \
    --train_episodes 1500 \
    --shared_encoder --freeze_all \
```

In the scripts, `env_name` is the control task type, including `ant`, `walker`, `cheetah`, and `hopper`. You can change the pomdp by replacing `pybullet_p` with `pybullet_v`. and other sequence model in `pomdp-continuous/configs/seq_models/`.

## Random Frame Dropping Task

After installing the packages and data, you can run the following script to reproduce results: 

```bash
cd defog
python main.py env=hopper model=dt
```

You can replace `hopper` with `halfcheetah`, `walker2d`. You can also replace `dt` with `dlstm` or `dlru` to test more sequence model.

# 🙏 Acknowledgement

The code is largely based on prior works:

- [Memory-RL](https://github.com/twni2016/Memory-RL)
- [DeFog](https://github.com/hukz18/DeFog)
- [LRURec](https://github.com/yueqirex/LRURec)
- [Minimal-LRU](https://github.com/NicolasZucchet/minimal-LRU)

# 🏷️ License

This work is licensed under the MIT license. See the [LICENSE](LICENSE) file for details.

# 📝 Citation

If you find our work useful, please consider citing:

```
@article{Lu2024Rethink,
  title={Rethinking Transformers in Solving POMDPs},
  author={Chenhao Lu and Ruizhe Shi and Yuyao Liu and Kaizhe Hu and Simon S. Du and Huazhe Xu},
  journal={International Conference on Machine Learning}, 
  year={2024}
}
```