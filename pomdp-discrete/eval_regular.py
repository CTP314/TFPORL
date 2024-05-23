import os, time

t0 = time.time()
pid = str(os.getpid())
if "SLURM_JOB_ID" in os.environ:
    jobid = str(os.environ["SLURM_JOB_ID"])
else:
    jobid = pid

import numpy as np
import torch
from absl import app, flags
from ml_collections import config_flags
import pickle
from utils import system, logger

from torchkit.pytorch_utils import set_gpu_mode
from policies.learner_regular import LearnerRegular
from envs.make_env import make_env

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config_env",
    None,
    "File path to the environment configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_rl",
    None,
    "File path to the RL algorithm configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_seq",
    "configs/seq_models/mlp_default.py",
    "File path to the seq model configuration.",
    lock_config=False,
)

# flags.mark_flags_as_required(["config_rl", "config_env"])

# shared encoder settings
flags.DEFINE_boolean("shared_encoder", False, "share encoder in actor-critic or not")
flags.DEFINE_boolean(
    "freeze_critic", False, "in shared encoder, freeze critic params in actor loss"
)

# training settings
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("batch_size", 64, "Mini batch size.")
flags.DEFINE_integer("train_episodes", 1000, "Number of episodes during training.")
flags.DEFINE_float("updates_per_step", 0.25, "Gradient updates per step.")
flags.DEFINE_integer("start_training", 10, "Number of episodes to start training.")

# logging settings
flags.DEFINE_boolean("debug", False, "debug mode")
flags.DEFINE_string("save_dir", "logs", "logging dir.")
flags.DEFINE_string("submit_time", None, "used in sbatch")
flags.DEFINE_string("run_name", None, "used in sbatch")

flags.DEFINE_string("load", None, "load model")


def load_flags_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        loaded_flags = pickle.load(f)
    return loaded_flags

def set_flags_value(f, k, v):
    if '.' in k:
        try:
            f_next = f[k.split('.')[0]].value
        except:
            f_next = f[k.split('.')[0]]
        set_flags_value(f_next, '.'.join(k.split('.')[1:]), v)
    else:
        setattr(f, k, v)

def main(argv):
    assert FLAGS.load is not None
    loaded_flags = load_flags_from_pickle(os.path.join(FLAGS.load, 'flags.pkl'))
    for k, v in loaded_flags['config_env'].items():
        set_flags_value(FLAGS.config_env, k, v)
    for k, v in loaded_flags['config_rl'].items():
        set_flags_value(FLAGS.config_rl, k, v)
    for k, v in loaded_flags['config_seq'].items():
        set_flags_value(FLAGS.config_seq, k, v)
    if FLAGS.config_env.env_type == 'regular_s5_fixed':
        FLAGS.config_env.env_type = 'regular_s5'
    for k, v in loaded_flags.items():
        if 'config_env.' in k:
            set_flags_value(FLAGS['config_env'].value, k.split('config_env.')[1], v)
        elif 'config_rl.' in k:
            set_flags_value(FLAGS['config_rl'].value, k.split('config_rl.')[1], v)
        elif 'config_seq.' in k:
            set_flags_value(FLAGS['config_seq'].value, k.split('config_seq.')[1], v)
        
    if FLAGS.seed < 0:
        seed = int(pid)  # to avoid conflict within a job which has same datetime
    else:
        seed = FLAGS.seed

    config_env = FLAGS.config_env
    config_rl = FLAGS.config_rl
    config_seq = FLAGS.config_seq

    config_env, env_name = config_env.create_fn(config_env)
    env = make_env(env_name, seed)
    eval_env = make_env(env_name, seed + 42, eval=True)

    system.reproduce(seed)
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)
    set_gpu_mode(torch.cuda.is_available())

    ## now only use env and time as directory name
    run_name = f"{config_env.env_type}/{config_env.env_name}/{config_seq.model.seq_model_config.name}/"
    config_seq, _ = config_seq.name_fn(config_seq, env.max_episode_steps)
    max_training_steps = int(FLAGS.train_episodes * env.max_episode_steps)
    config_rl, _ = config_rl.name_fn(
        config_rl, env.max_episode_steps, max_training_steps
    )
    format_strs = ["csv"]

    format_strs.extend(["stdout", "log"])
    uid = FLAGS.load.split('/')[-1]
    run_name += uid

    log_path = os.path.join(FLAGS.save_dir, run_name)
    logger.configure(dir=log_path, format_strs=format_strs, log_suffix="_eval")

    # start training
    learner = LearnerRegular(env, eval_env, FLAGS, config_rl, config_seq, config_env)
    learner.load_model(system.get_ckpt_path(log_path))
    # print(learner.evaluate_on_len(deterministic=True))
    learner.log_on_len()

if __name__ == "__main__":
    app.run(main)