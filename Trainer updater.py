#Kidus Berhanu
import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        if self.config.ckpt_path is not None:
            ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
            logger.info("saving %s", self.config.ckpt_path)
            torch.save(ckpt_model.state_dict(), self.config.ckpt_path)

    def run_epoch(self, split):
        is_train = split == 'train'
        self.model.train(is_train)
        data = self.train_dataset if is_train else self.test_dataset
        loader = DataLoader(data, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

        losses = []
        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        for it, (x, y) in pbar:

            # place data on the correct device
            x = x.to(self.device)
            y = y.to(self.device)

            # forward the model
            with torch.set_grad_enabled(is_train):
                logits, loss = model(x, y)
