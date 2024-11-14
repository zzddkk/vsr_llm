import torch
import os
import hydra
from vsr_llm_dataset import video_dataset
from torch.nn.utils.rnn import pad_sequence
from transform import VideoTransform
from samplers import (
    ByFrameCountSampler,
    DistributedSamplerWrapper,
    RandomSamplerWrapper,
)
def collater(samples):
    Batch={}
    input_lengths = torch.tensor([sample["video"].size(0) for sample in samples],dtype=torch.long)
    Batch["video"] = pad_sequence([sample["video"] for sample in samples],padding_value=0,batch_first=True)
    Batch["target"] = [sample["target"] for sample in samples]
    Batch["input_lengths"] = input_lengths
    Batch["file_path"] = [sample["file_path"] for sample in samples]
    return Batch

class DataModule():
    def __init__(self, cfg=None):
        self.cfg = cfg
        # self.cfg.gpus = torch.cuda.device_count()
        # self.total_gpus = self.cfg.gpus * self.cfg.trainer.num_nodes

    def _dataloader(self, ds, sampler ,collate_fn):
        return torch.utils.data.DataLoader(
            ds,
            num_workers=8,
            pin_memory=True,
            batch_sampler=sampler,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        ds_args = self.cfg.dataset
        train_ds = video_dataset(
            root_dir=ds_args.root_dir,
            subset="train",
            video_transform=VideoTransform("train"),
        )
        sampler = ByFrameCountSampler(train_ds, self.cfg.dataset.max_frames)
        return self._dataloader(train_ds, sampler ,collater)

    def val_dataloader(self):
        ds_args = self.cfg.dataset
        val_ds = video_dataset(
            root_dir=ds_args.root_dir,
            subset="val",
            video_transform=VideoTransform("val"),
        )
        sampler = ByFrameCountSampler(val_ds, self.cfg.dataset.max_frames,shuffle=False)
        return self._dataloader(val_ds, sampler , collater)

    def test_dataloader(self):
        ds_args = self.cfg.dataset
        dataset = video_dataset(
            root_dir=ds_args.root_dir,
            subset="test",
            video_transform=VideoTransform("test"),
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,collate_fn=collater,num_workers=12)
        return dataloader

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
@hydra.main(config_path=os.path.join(parent_dir,"conf"), config_name="configs")
def test(cfg):
    from transform import VideoTransform
    datamodel = DataModule(cfg)
    val_dataloader = datamodel.val_dataloader()
    for batch in val_dataloader:
        print(batch["video"].shape)
        break
if __name__ == "__main__":
    test()