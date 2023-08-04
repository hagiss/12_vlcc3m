import os
import numpy as np
import math
import shutil
import argparse
import importlib
import torch
import transformers
from transformers import HfArgumentParser, TrainingArguments, AutoTokenizer, AutoProcessor, CLIPImageProcessor
from torch.optim import AdamW
from dataclasses import dataclass, field
from typing import Any, Optional, Union, Dict, Sequence, List
from tqdm import tqdm
import webdataset as wds
import braceexpand
from model import Clip_FlanT5
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import torch.optim.lr_scheduler as lr_scheduler
import wandb
import pdb
from peft import get_peft_model_state_dict, PeftModel
from itertools import islice


@dataclass
class DataCollatorForVisionLanguageDataset(object):
    """Collate examples for vision-language fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    processor: transformers.CLIPImageProcessor

    def __call__(self, batch):
        image_inputs = self.processor(images=batch[0], return_tensors="pt")
        text_inputs = self.tokenizer(batch[1], return_tensors="pt", padding=True, truncation=True)

        return {**image_inputs, **text_inputs}
        

@dataclass
class DefaultTrainingArguments(TrainingArguments):
    # data related
    dataset_path: str = field(default="/data/private/cc3m")
    cache_dir: Optional[str] = field(default="/data/private/cc3m/cache", metadata={"help": "Cache directory for the datasets."})

    # model related
    vision_model_name_or_path: str = field(default="openai/clip-vit-base-patch32")
    text_model_name_or_path: str = field(default="google/flan-t5-large")

    lora_dropout: float = field(default=0.1)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=128)
    use_lora: bool = field(default=True)

    num_learnable_tokens: int = field(default=32)

    # training related
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=128)
    per_device_eval_batch_size: int = field(default=128)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=1e-4)
    warmup_ratio: float = field(default=.1)
    dataloader_num_workers: int = field(default=4)

    # logging related
    output_dir: str = field(default="/data/private/cc3m/output")
    project_name: str = field(default="vl")


class WarmUpCosineAnnealingLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, T_max, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            annealing_phase = self.last_epoch - self.warmup_steps
            T_max = self.T_max - self.warmup_steps
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * annealing_phase / T_max)) / 2
                    for base_lr in self.base_lrs]


class VL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Clip_FlanT5(args)

        self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = {
            'scheduler': WarmUpCosineAnnealingLR(self.optimizer, warmup_steps=int(self.args.training_steps*self.args.warmup_ratio), T_max=self.args.training_steps),
            'interval': 'step',
            'frequency': 1
        }

        self.validation_step_outputs = []

    def generate(self, pixel_values, input_ids=None, attention_mask=None):
        return self.model.generate(input_ids, attention_mask, pixel_values)

    def forward(self, input_ids, attention_mask, pixel_values):
        return self.model(input_ids, attention_mask, pixel_values)
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs
        self.log('train_loss', loss.detach().cpu() , on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.detach().cpu()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.validation_step_outputs.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.validation_step_outputs).mean()
        self.validation_step_outputs = []
        return avg_val_loss

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        del checkpoint['state_dict']

        checkpoint['state_dict'] = self.model.vision_projector.state_dict()
        if self.args.use_lora:
            path = self.trainer.checkpoint_callback.dirpath + "/adapter"
            checkpoint['adapter'] = path
            self.model.text_decoder.save_pretrained(path)


def main():
    parser = HfArgumentParser(DefaultTrainingArguments)
    args, = parser.parse_args_into_dataclasses()

    # load datasets
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name_or_path, cache_dir=args.cache_dir)
    processor = AutoProcessor.from_pretrained(args.vision_model_name_or_path, cache_dir=args.cache_dir).image_processor

    val_dataset_size = 12808
    # train_dataset_size = 12808
    train_dataset_size = 258856
    # train_dataset_size = 2770590

    num_gpus = torch.cuda.device_count()
    args.training_steps = args.num_train_epochs * math.ceil(train_dataset_size / (num_gpus * args.per_device_train_batch_size * args.gradient_accumulation_steps))

    dataset_files_train = list(braceexpand.braceexpand(args.dataset_path + "/cc3m_train/{00000..00030}.tar"))
    dataset_files_val = list(braceexpand.braceexpand(args.dataset_path + "/cc3m_val/{00000..00001}.tar"))
    # dataset_files_train = list(braceexpand.braceexpand(args.dataset_path + "/cc3m_val/{00000..00001}.tar"))

    val_dataset = (
        wds.WebDataset(dataset_files_val)
        .decode("pil")
        .to_tuple("jpg;png", "txt")
        .batched(args.per_device_eval_batch_size, partial=False)
        # .with_epoch(val_dataset_size // args.per_device_eval_batch_size)
    )
    train_dataset = (
        wds.WebDataset(dataset_files_train)
        .shuffle(100)
        .decode("pil")
        .to_tuple("jpg;png", "txt")
        .batched(args.per_device_train_batch_size, partial=False)
        # .with_epoch(train_dataset_size // args.per_device_train_batch_size)
    )

    collate_fn = DataCollatorForVisionLanguageDataset(tokenizer, processor)

    val_loader = wds.WebLoader(val_dataset, num_workers=2, collate_fn=collate_fn, pin_memory=True, batch_size=None)
    train_loader = wds.WebLoader(train_dataset, num_workers=args.dataloader_num_workers, collate_fn=collate_fn, pin_memory=True, batch_size=None)
    val_loader.length = val_dataset_size // args.per_device_eval_batch_size
    train_loader.length = train_dataset_size // args.per_device_train_batch_size

    vl_model = VL(args)

    logger = WandbLogger(project=args.project_name, log_model=True, save_code=True, name="clip-flan-t5", settings=wandb.Settings(start_method='fork'))
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # load pl trainer
    trainer = pl.Trainer(
        devices=num_gpus,
        accelerator="gpu",
        strategy="ddp",
        max_epochs=args.num_train_epochs,
        max_steps=args.training_steps,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=1.0,
        logger=logger,
        log_every_n_steps=5,
        num_sanity_val_steps=0,
        # val_check_interval=train_loader.length // 10,
        default_root_dir=args.output_dir,
        callbacks=[lr_monitor]
    )

    trainer.fit(vl_model, train_loader, val_loader)
   

if __name__ == '__main__':
    main()
