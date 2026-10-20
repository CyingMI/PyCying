import sys
import os

sys.path.append('..')

import torch
from cying import nn as cynn
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, IterableDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from torchinfo import summary
num_class = 2*10**4
opt_lang_model = cynn.LangModel(
    vocab_size = num_class,
    d_model = 1024,
    num_heads = 16,
    hidden_width = 4096,
    num_layers = 16,
    padding_idx = 0
).to(device)
summary(opt_lang_model)

import json
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader

max_seq_len = 1024
batch_size = 10
grad_accum_steps = 32

data_path = f'/root/autodl-tmp/hf_sample_200G'
tokenizer_path = f"./cci3_tokenizer.json"

class CCI3(IterableDataset):
    def __init__(self, data_path, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = load_dataset(
            data_path,
            split="train",
            streaming=True,
        ).shuffle(seed=42, buffer_size=100)

    def __iter__(self):
        for sample in self.dataset:
            encoding = self.tokenizer.encode(sample["text"])

            input_ids = torch.tensor(encoding.ids, dtype=torch.long)
            attention_mask = torch.tensor(
                encoding.attention_mask, dtype=torch.long
            )
            labels = input_ids.clone()

            yield input_ids[:-1], labels[1:]


tokenizer = Tokenizer.from_file(tokenizer_path)
tokenizer.enable_truncation(max_length=max_seq_len)
tokenizer.enable_padding(length=max_seq_len, pad_id=0)

dataset = CCI3(data_path, tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size)

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

lr = 1e-4
optim = torch.optim.AdamW(opt_lang_model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
scheduler = CosineAnnealingWarmRestarts(optim, T_0=400, T_mult=2, eta_min=1e-6)

k = 1000

from torch.amp import autocast, GradScaler

scaler = GradScaler()

pbar = tqdm(dataloader)

loss_epoch = []
i = 1
accum_count = 0

for token_ids, tgt_ids in pbar:
    opt_lang_model.train()
    token_ids = token_ids.to(device)
    tgt_ids = tgt_ids.to(device)

    with autocast(device_type="cuda", dtype=torch.float16):
        outputs = opt_lang_model(
            token_seq=token_ids,
            causal_mask=torch.triu(
                torch.ones(max_seq_len-1, max_seq_len-1), diagonal=1
            ).bool().to(device)
        )
        loss = criterion(
            outputs.view(-1, outputs.size(-1)),
            tgt_ids.reshape(-1)
        )
        loss = loss / grad_accum_steps

    scaler.scale(loss).backward()
    accum_count += 1

    if accum_count == grad_accum_steps:
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()
        scheduler.step()
        accum_count = 0

    loss_epoch.append(loss.item() * grad_accum_steps)
    pbar.set_description(f"Loss {loss.item()*grad_accum_steps:.4e}")

    if i % k == 0:
        opt_lang_model.eval()
        torch.save(opt_lang_model.state_dict(), f'./model.pth')
        torch.save(torch.tensor(loss_epoch), f'./loss_values.pth')
        i = 1
    else:
        i += 1