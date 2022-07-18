import logging
import os

from torch.utils.data import DataLoader

import wandb
from src.config import Config
from src.dataset import LayoutDataset
from src.logger import Logger
from src.trainer import Trainer
from src.utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

if __name__ == "__main__":
    logger = Logger()
    config = Config()

    logging.info("start testing placement model!!!!")
    set_seed(config.args.seed)

    test_set = LayoutDataset(config, split="test")
    test_loader = DataLoader(test_set, batch_size=config.args.batch_size, shuffle=False, num_workers=config.args.num_workers)
    logging.info(f"datasets loaded, test: {len(test_set)}")

    trainer = Trainer(config)
    trainer.test(test_loader)

