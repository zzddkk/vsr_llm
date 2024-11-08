import hydra
from omegaconf import OmegaConf
@hydra.main(config_path="conf", config_name="configs")
def train(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    train()