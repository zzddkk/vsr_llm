import hydra
import os
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf
from Trainer import Trainer
from utils import Logger ,set_seed,check_ckpt_path
from vsr_llm_datamodule import DataModule
from vsr_llm import ModelModule

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
@hydra.main(version_base="1.3",config_path=os.path.join(parent_dir,"conf"), config_name="configs")
def train(cfg) -> None:
    # print(OmegaConf.to_yaml(cfg))
    # Initialize logger
    logger = Logger(cfg.log_dir)
    # Set seed
    set_seed(1337)
    # Initialize accelerator


    project_config = ProjectConfiguration(automatic_checkpoint_naming=True,total_limit=cfg.trainer.total_limit,save_on_each_node=False,project_dir=cfg.ckpt_path)
    accelerator = Accelerator(gradient_accumulation_steps = cfg.trainer.gradient_accumulation_steps,project_config=project_config,mixed_precision=cfg.trainer.mixed_precision)
    accelerator.even_batches=False
    datamodule = DataModule(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelmodule = ModelModule(cfg)

    trainer = Trainer(cfg, modelmodule, datamodule,device,logger,accelerator)
    with accelerator.autocast():
        trainer.train()
    # trainer.test()
    # alter the ckpt save path to per training to ensure the ckpt path is correct
    path = check_ckpt_path(cfg.ckpt_path)

if __name__ == "__main__":
    train()