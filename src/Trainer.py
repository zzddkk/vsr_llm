import accelerate
import torch
from tqdm import tqdm
from accelerate.utils import find_executable_batch_size
from torcheval.metrics import WordErrorRate
from utils import get_cosine_schedule_with_warmup,check_ckpt_path
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

    # This decorator will find the maximum batch size that can be used for training
    @find_executable_batch_size(starting_batch_size=4)
    def train(batch_size,self):
        self.train_dataloader = self.datamodule.train_dataloader(batch_size)
        # Train batch size:
        self.accelerator.print(f"Train batch size: {batch_size* self.accelerator.num_processes*self.cfg.trainer.gradient_accumulation_steps}")
        self.val_dataloader = self.datamodule.val_dataloader(batch_size)
        self.test_dataloader = self.datamodule.test_dataloader()
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.cfg, int(len(self.train_dataloader) * self.cfg.trainer.epochs / self.cfg.trainer.gradient_accumulation_steps))
        """
        You only need to use prepare once. If you saved the unwrapped model given by the accelerator.unwrap_model(model) method, 
        you should first load your model then send it to prepare. 
        If you saved the wrapped model, you should first send it to prepare then load your checkpoint.
        """
        self.model,self.optimizer ,self.scheduler,self.train_dataloader,self.val_dataloader,self.test_dataloader= self.accelerator.prepare(self.model,self.optimizer,self.scheduler,self.train_dataloader,self.val_dataloader,self.test_dataloader)
        self.accelerator.register_for_checkpointing(self.scheduler)
        if self.cfg.resume_from_checkpoint!='None':
            self.accelerator.load_state(self.cfg.resume_from_checkpoint)
        for epoch in range(self.cfg.trainer.epochs):
            self.epoch = epoch
            self._inner_training_loop()
            self.accelerator.save_state()
            self._inner_validation_loop()
        # self._inner_test_loop()
        self.accelerator.save_model(self.model,self.cfg.ckpt_path)
        self.logger.close()

    def test(self):
        self.model.load_state_dict(torch.load(self.cfg.ckpt_path)["model"],strict=False)
        self.test_dataloader = self.datamodule.test_dataloader()
        self.model,self.test_dataloader = self.accelerator.prepare(self.model,self.test_dataloader)
        self._inner_test_loop()
        self.logger.close()
    
    # inner training loop
    def _inner_training_loop(self):
        self.model.train()
        pbar = tqdm(self.train_dataloader, desc=f"Training {self.epoch +1 }/{self.cfg.trainer.epochs}", position=0, leave=True)
        step = 0
        for batch in pbar:
            with self.accelerator.accumulate():
                loss = self.modelmodule.training_step(batch)
                if step % self.cfg.trainer.gradient_accumulation_steps == 0:
                    self.logger.add_scalar("loss/train",loss.item())
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_value_(self.model.parameters(), self.cfg.trainer.clip_value)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                step += 1
            pbar.set_postfix({"loss":loss.item()})

    # inner validation loop
    def _inner_validation_loop(self):
        self.model.eval()
        pbar = tqdm(self.val_dataloader, desc="Validation", position=0, leave=True)
        loss_record = []
        for batch in pbar:
            loss = self.modelmodule.val_step(batch)
            loss_record.append(loss.item())
            pbar.set_postfix({"loss":loss.item()})
        self.logger.add_scalar("loss/val",sum(loss_record)/len(loss_record))
    
    # test loop
    def _inner_test_loop(self):
        self.metric.reset()
        self.model.eval()
        pbar = tqdm(self.test_dataloader, desc="Testing", position=0, leave=True)
        for batch in pbar:
            pre = self.modelmodule.test_step(batch)
            self.metric.update([pre[0],batch["target"][0]])
            self.logger.write(f"{pre[0]},{batch['target'][0]},{batch['file_path'][0]}")
        wer = self.metric.compute()
        self.accelerator.print(f"Word Error Rate: {wer}")
            