import hydra
import os
import torch
import shutil
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

    # Set seed
    set_seed(1337)
    # Initialize accelerator


    project_config = ProjectConfiguration(automatic_checkpoint_naming=True,total_limit=cfg.trainer.total_limit,save_on_each_node=False,project_dir=cfg.ckpt_path)
    accelerator = Accelerator(gradient_accumulation_steps = cfg.trainer.gradient_accumulation_steps,project_config=project_config,mixed_precision=cfg.trainer.mixed_precision)
    accelerator.even_batches=False
    # Initialize logger
    if accelerator.is_main_process:
        logger = Logger(cfg.log_dir,"train")
    else:
        logger = None
    datamodule = DataModule(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelmodule = ModelModule(cfg)

    trainer = Trainer(cfg, modelmodule, datamodule,device,logger,accelerator)
    with accelerator.autocast():
        trainer.train()
    # trainer.test()
    # alter the ckpt save path to per training to ensure the ckpt path is correct
    
    if accelerator.is_main_process:
        path = check_ckpt_path(cfg.ckpt_path)
        logger.save_parmas(path,cfg)
        logger.move_log("train.log",parent_dir)
        shutil.move(os.path.join(cfg.ckpt_path,"pytorch_model.bin"),os.path.join(path,"pytorch_model.bin"))
        logger.close()
if __name__ == "__main__":
    train()