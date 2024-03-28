
import hydra
from omegaconf import DictConfig, OmegaConf

import os

from experiment import run_experiment


@hydra.main(version_base="1.3.2", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    
    

    # Open up the configuration to allow changes
    #OmegaConf.set_struct(cfg, False)

     # Merge 'experiment' and 'model' sections of the config
    #merged_cfg = OmegaConf.merge(cfg.experiment, cfg.model)

    # Call run_experiment with the merged configuration
    #run_experiment(**merged_cfg)

    run_experiment(cfg)


if __name__ == "__main__":
    main()