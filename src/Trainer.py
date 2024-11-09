import accelerate
import tqdm
from accelerate.utils import find_executable_batch_size
from torcheval.metrics import WordErrorRate
class Trainer:
    def __init__(self, cfg ,modelmodule,datamodule ,optimizer, scheduler,device,logger,accelerator):
        self.cfg = cfg
        self.modelmoudule = modelmoudule
        self.model = self.modelmodule.build_model(cfg)
        self.optimizer = optimizer
        self.device = device
        self.datamodule = datamodule
        self.scheduler = scheduler
        self.metric = WordErrorRate()
        self.logger = logger

    @find_executable_batch_size(starting_batch_size=128)
    def train(self,batch_size):
        for epoch in range(self.cfg.trainer.epochs):
            self._inner_training_loop(batch_size)
            self._inner_validation_loop(batch_size)
    def _inner_training_loop(self):
        train_dataloader = self.datamodule.train_dataloader()
        self.model.train()
        pbar = tqdm(train_dataloader, desc="Training", position=0, leave=True)
        for batch in pbar:
            with accelerate.accumulate():
                loss = self.modelmoudule.training_step(batch)
                accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            pbar.set_postfix({"loss":loss.item()})
    def _inner_validation_loop(self):
        val_dataloader = self.datamodule.val_dataloader()
        self.model.eval()
        pbar = tqdm(val_dataloader, desc="Validation", position=0, leave=True)
        for batch in pbar:
            loss = self.modelmoudule.validation_step(batch)
            pbar.set_postfix({"loss":loss.item()})
    def _inner_test_loop(self):
        self.metric.reset()
        test_dataloader = self.datamodule.test_dataloader()
        self.model.eval()
        pbar = tqdm(test_dataloader, desc="Testing", position=0, leave=True)
        for batch in pbar:
            pre = self.modelmoudule.test_step(batch)
            self.metric.update([pre,batch["target"]])
        wer = self.metric.compute()
            