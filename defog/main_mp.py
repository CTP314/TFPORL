import os
import hydra
import torch
import utils
import logging
from utils import config_logging, get_log_dict
from core import train
import torch.multiprocessing as mp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
# torch.set_float32_matmul_precision('high')

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    manager = mp.Manager()
    num_seeds = len(cfg.seeds)
    barrier = manager.Barrier(num_seeds)
    log_dict = get_log_dict(manager, num_seeds)
    dataset_dir = os.path.join(os.path.expanduser("~"), ".d4rl")
    pool = mp.Pool(num_seeds)
    pool.starmap(train, [(cfg, seed, log_dict, idx, logger, barrier, dataset_dir) for (idx, seed) in enumerate(cfg.seeds)])
    pool.close()
    pool.join()
    utils.visualize_perf_drop_curve(cfg, log_dict)

if __name__ == "__main__":
    mp.set_start_method('spawn')  # set spawn for linux servers
    main()