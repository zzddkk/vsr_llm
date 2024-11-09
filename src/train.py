import hydra
import os
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from Trainer import Trainer
from utils import Logger ,set_seed
from vsr_llm_datamodule import DataModule
from vsr_llm import ModelModule

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
@hydra.main(config_path=os.path.join(parent_dir,"conf"), config_name="configs")
def train(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))
    logger = Logger(cfg.log_dir)
    set_seed(1337)
    accelerator = Accelerator(gradient_accumulation_steps = cfg.trainer.gradient_accumulation_steps)
    datamodule = DataModule(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelmodule = ModelModule(cfg)

    trainer = Trainer(cfg, modelmodule, datamodule,device,logger,accelerator)
    trainer.train()

if __name__ == "__main__":
    train()