import hydra
import os
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf
from Trainer import Trainer
from utils import Logger ,set_seed
from vsr_llm_datamodule import DataModule
from vsr_llm import ModelModule

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
@hydra.main(config_path=os.path.join(parent_dir,"conf"), config_name="configs")
def test(cfg) -> None:
    # print(OmegaConf.to_yaml(cfg))
    # Set seed
    set_seed(1337)
    # Initialize accelerator
    project_config = ProjectConfiguration(automatic_checkpoint_naming=True,total_limit=cfg.trainer.total_limit,save_on_each_node=False)
    accelerator = Accelerator(gradient_accumulation_steps = cfg.trainer.gradient_accumulation_steps,project_config=project_config,mixed_precision=cfg.trainer.mixed_precision)
    datamodule = DataModule(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelmodule = ModelModule(cfg)
    # Initialize logger
    if accelerator.is_main_process:
        logger = Logger(cfg.log_dir,"eval")
    else:
        logger = None

    trainer = Trainer(cfg, modelmodule, datamodule,device,logger,accelerator)
    trainer.test()
    if accelerator.is_main_process:
        logger.move_log("eval.log",parent_dir)
        logger.close()

if __name__ == "__main__":
    test()