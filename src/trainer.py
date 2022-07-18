import logging
import os

import torch
import torch.nn as nn
import wandb
from tokenization_layout import T5LayoutTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import (AdamW, T5Config, T5ForConditionalGeneration,
                          T5Tokenizer, get_linear_schedule_with_warmup)

from .config import Config
from .utils import AverageMeter


class Trainer:
    def __init__(self, config: Config, device=None, world_size=None):
        self.config = config
        if config.args.model_name_or_path is not None:
            self.model = T5ForConditionalGeneration.from_pretrained(config.args.model_name_or_path)
        elif config.args.config is not None:
            t5_config = T5Config.from_pretrained(config.args.config)
            self.model = T5ForConditionalGeneration(t5_config)
        else:
            logging.error("set eithor `config` or `model_name_or_path`")
        self.tokenizer = T5Tokenizer.from_pretrained(config.args.tokenizer_path)
        self.device = device
        self.world_size = world_size

    def train(self, train_loader, valid_loader):
        num_train = len(train_loader.dataset) // self.world_size + 1
        num_valid = len(valid_loader.dataset)
    
        total_steps = (num_train // self.config.args.batch_size) * self.config.args.epochs if num_train % self.config.args.batch_size == 0 \
            else (num_train // self.config.args.batch_size + 1) * self.config.args.epochs

        self.model.to(self.device)
        if self.config.args.distribute:
            self.model = DDP(self.model, device_ids=[self.config.args.local_rank])

        optimizer = AdamW(self.model.parameters(), lr=self.config.args.lr, weight_decay=self.config.args.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.config.args.warmup_ratio*total_steps,
                                                    num_training_steps=total_steps)

        if self.config.args.local_rank == 0 or self.config.args.local_rank == -1:
            logging.info("-------------start training-------------")
        step = 0
        for epoch in range(self.config.args.epochs):
            if self.config.args.local_rank == 0 or self.config.args.local_rank == -1:
                logging.info(f"-----epoch {epoch}------")
            self.model.train()
            with torch.enable_grad(), tqdm(total=(num_train // self.config.args.batch_size + 1)) as progress_bar:
                for _, batch in enumerate(train_loader):
                    step += 1
                    source_text = batch['source_text']
                    target_text = batch['target_text']
                    batch_size = len(source_text)
                    source_dict = self.tokenizer(source_text, padding='longest', truncation=True, max_length=512, return_tensors='pt')
                    target_dict = self.tokenizer(target_text, padding='longest', truncation=True, max_length=512, return_tensors='pt')

                    input_ids = source_dict['input_ids'].to(self.device, dtype=torch.long)
                    attention_mask = source_dict['attention_mask'].to(self.device, dtype=torch.long)
                    target_ids = target_dict['input_ids'].to(self.device, dtype=torch.long)
                    decoder_attention_mask = target_dict['attention_mask'].to(self.device, dtype=torch.long)
                    target_ids[target_ids[:, :] == 0] = -100
                    label_ids = target_ids.to(self.device)

                    res = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids, decoder_attention_mask=decoder_attention_mask, return_dict=True)
                    loss, logits = res['loss'], res['logits']
                    # loss = loss.mean()
                    loss_val = loss.item()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()

                    progress_bar.update(1)
                    progress_bar.set_postfix(epoch=epoch, loss=loss_val)

                    if self.config.args.local_rank == 0 or self.config.args.local_rank == -1:
                        wandb.log({'training loss': loss_val}, step)
                        wandb.log({'training_lr': optimizer.param_groups[0]['lr']}, step)
            

            if self.config.args.local_rank == 0 or self.config.args.local_rank == -1:
                logging.info(f'------evaluation at epoch {epoch}------')
                self.model.eval()

                loss_meter = AverageMeter()
                with torch.no_grad(), tqdm(total=num_valid) as progress_bar:
                    for _, batch in enumerate(valid_loader):
                        source_text = batch['source_text']
                        target_text = batch['target_text']
                        batch_size = len(source_text)
                        source_dict = self.tokenizer(source_text, padding='longest', truncation=True, max_length=512, return_tensors='pt')
                        target_dict = self.tokenizer(target_text, padding='longest', truncation=True, max_length=512, return_tensors='pt')

                        input_ids = source_dict['input_ids'].to(self.device, dtype=torch.long)
                        attention_mask = source_dict['attention_mask'].to(self.device, dtype=torch.long)
                        target_ids = target_dict['input_ids'].to(self.device, dtype=torch.long)
                        decoder_attention_mask = target_dict['attention_mask'].to(self.device, dtype=torch.long)
                        target_ids[target_ids[:, :] == 0] = -100
                        label_ids = target_ids.to(self.device)

                        res = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids, decoder_attention_mask=decoder_attention_mask, return_dict=True)
                        loss, logits = res['loss'], res['logits']
            
                        loss_val = loss.item()
                        loss_meter.update(loss_val, batch_size)

                        progress_bar.update(batch_size)
                        progress_bar.set_postfix(NLL=loss_meter.avg)
                    wandb.log({'valid loss': loss_meter.avg}, step)
                
                self.model.module.save_pretrained(f"./{self.config.args.output_dir}/epoch-{epoch}-loss-{loss_meter.avg}")
                self.tokenizer.save_pretrained(f"./{self.config.args.output_dir}/epoch-{epoch}-loss-{loss_meter.avg}")

    def test(self, test_loader):
        pred_list = []
        label_list = []
        num_test = len(test_loader.dataset)
        self.model.to(self.device)
        logging.info("-------------start testing-------------")
        self.model.eval()

        with torch.no_grad(), tqdm(total=num_test) as progress_bar:
            for _, batch in enumerate(test_loader):
                source_text = batch['source_text']
                target_text = batch['target_text']
                batch_size = len(source_text)
                source_dict = self.tokenizer(source_text, padding='longest', truncation=True, return_tensors='pt')

                input_ids = source_dict['input_ids'].to(self.device, dtype=torch.long)
                attention_mask = source_dict['attention_mask'].to(self.device, dtype=torch.long)

                generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, eos_token_id=1, max_length=200)
                # # ------
                # print(generated_ids)
                # exit()
                # # ------
                outputs_decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                pred_list.extend(outputs_decoded)
                label_list.extend(target_text)
                progress_bar.update(batch_size)
        
                with open(os.path.join(self.config.args.model_name_or_path, "prediction.txt"), "a") as f:
                    for i in range(len(outputs_decoded)):
                        f.write(outputs_decoded[i] + '\n')