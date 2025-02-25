import os

import hydra
import lightning as L
import lovely_tensors as lt
import rootutils
import torch
from beartype.typing import Any, Dict, List, Optional, Tuple
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies.strategy import Strategy
from omegaconf import DictConfig

lt.monkey_patch()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from flowdock import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from flowdock import register_custom_omegaconf_resolvers, resolve_omegaconf_variable
from flowdock.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)



@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    # return optimized metric
    return metric_value


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
