import os
import hydra
import wandb

from omegaconf import DictConfig, OmegaConf


from cv_experiment import run_experiment


@hydra.main(version_base="1.3.2", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    
    os.environ["WANDB_MODE"] = "offline"

    print("start")
    print(cfg)
    

    run_experiment(cfg)

    


if __name__ == "__main__":
    main()