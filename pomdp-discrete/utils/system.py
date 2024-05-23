import numpy as np
import random
import torch
import datetime
import dateutil.tz
import os


def reproduce(seed):
    """
    This can only fix the randomness of numpy and torch
    To fix the environment's, please use
        env.seed(seed)
        env.action_space.np_random.seed(seed)
    We have add these in our training script
    """
    assert seed >= 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def now_str():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime(
        "%Y-%m-%d-%H:%M:%S"
    )  # may cause collision, please use PID to prevent

def get_ckpt_path(log_path, step=None):
    # agent_00210_perf0.490
    ckpt_path = os.path.join(log_path, "save")
    ckpt_dict = {}
    for file in os.listdir(ckpt_path):
        if file.endswith(".pt") and not file.endswith("trajs.pt"):
            ckpt_dict[int(file.split("_")[1])] = file
    
    if step is None:
        step = max(ckpt_dict.keys())
    else:
        assert step in ckpt_dict.keys()
        
    return os.path.join(ckpt_path, ckpt_dict[step])