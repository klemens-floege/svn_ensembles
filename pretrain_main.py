import os
import hydra
import wandb

from omegaconf import DictConfig, OmegaConf


from pretrain_experiment import pretrain_experiment


@hydra.main(version_base="1.3.2", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    
    os.environ["WANDB_MODE"] = "offline"
    #os.environ["WANDB_DIR"] = cfg.experiment.wandb_path

    #config = ExperimentConfig(**exp_dict, save_folder = SAVE_FOLDER)
    #config.experiment.save_path = ''

    # Your code here using cfg for accessing configurations
    print("start")
    print(cfg)
    

    pretrain_experiment(cfg)

    


if __name__ == "__main__":
    main()