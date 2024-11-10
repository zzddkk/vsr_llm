import accelerate
import torch
from tqdm import tqdm
from accelerate.utils import find_executable_batch_size
from torcheval.metrics import WordErrorRate
from utils import get_cosine_schedule_with_warmup
class Trainer:
    def __init__(self, cfg ,modelmodule,datamodule,device,logger,accelerator):
        self.cfg = cfg
        self.modelmodule = modelmodule
        self.model = self.modelmodule.build_model(cfg)
        self.optimizer = torch.optim.AdamW([{"name": "model", "params": self.model.parameters(), "lr": self.cfg.trainer.lr}], weight_decay=self.cfg.trainer.weight_decay, betas=(0.9, 0.98))
        self.device = device
        self.datamodule = datamodule
        self.accelerator = accelerator
        self.metric = WordErrorRate()
        self.logger = logger


    @find_executable_batch_size(starting_batch_size=16)
    def train(batch_size,self):
        self.train_dataloader = self.datamodule.train_dataloader(batch_size)
        self.val_dataloader = self.datamodule.val_dataloader(batch_size)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.cfg, int(len(self.train_dataloader) * self.cfg.trainer.epochs / self.cfg.trainer.gradient_accumulation_steps))
        self.model,self.optimizer ,self.scheduler,self.train_dataloader,self.val_dataloader = self.accelerator.prepare(self.model,self.optimizer,self.scheduler,self.train_dataloader,self.val_dataloader)
        for epoch in range(self.cfg.trainer.epochs):
            self.accelerator.free_memory()
            self._inner_training_loop()
            self._inner_validation_loop()
    def _inner_training_loop(self):
        self.model.train()
        pbar = tqdm(self.train_dataloader, desc="Training", position=0, leave=True)
        for batch in pbar:
            with self.accelerator.accumulate():
                loss = self.modelmodule.training_step(batch)
                self.logger.add_scalar("loss/train",loss.item())
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            pbar.set_postfix({"loss":loss.item()})
    def _inner_validation_loop(self):
        self.model.eval()
        pbar = tqdm(self.val_dataloader, desc="Validation", position=0, leave=True)
        for batch in pbar:
            loss = self.modelmodule.val_dataloader(batch)
            self.logger.add_scalar("loss/val",loss.item())
            pbar.set_postfix({"loss":loss.item()})
    def _inner_test_loop(self):
        self.test_dataloader = self.datamodule.test_dataloader()
        self.metric.reset()
        self.model.eval()
        pbar = tqdm(self.test_dataloader, desc="Testing", position=0, leave=True)
        for batch in pbar:
            pre = self.modelmodule.test_step(batch)
            self.metric.update([pre,batch["target"]])
        wer = self.metric.compute()
            