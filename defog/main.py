# TODO: test percentage buffer
# TODO: test performance control
import os
import hydra
import utils
import torch
import logging
from core import train
logger = logging.getLogger(__name__)
# torch.set_float32_matmul_precision('high')

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    dataset_dir = os.path.join(os.path.expanduser("~"), ".d4rl")
    log_dict = utils.get_log_dict()
    for seed in cfg.seeds:
        train(cfg, seed, log_dict, -1, logger, None, dataset_dir)
    utils.visualize_perf_drop_curve(cfg, log_dict)

if __name__ == "__main__":
    main()
