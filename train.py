import hydra
from accelerate import Accelerator
from omegaconf import OmegaConf
from src.Trainer import Trainer
from src.utils import Logger
from src.vsl_lm_datamodule import DataModule
from src.vsr_llm import ModelModule
@hydra.main(config_path="conf", config_name="configs")
def train(cfg) -> None:
    logger = Logger(cfg.log_dir)
    accelerator = Accelerator(gradient_accumulation_steps = cfg.trainer.gradient_accumulation_steps)
    datamodule = DataModule(cfg)
    modelmodule = ModelModule(cfg)
    trainer = Trainer(cfg, modelmodule, datamodule,optmizer,device,logger,accelerator)
    train.train()

if __name__ == "__main__":
    train()