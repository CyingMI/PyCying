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

scale = 1
max_seq_len = 256
batch_size = 24

data_path = f'/root/autodl-tmp/data'
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

tokenizer.enable_padding(
    length=max_seq_len,
    pad_id=0
)

dataset = CCI3(
    data_path,
    tokenizer
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size
)

from tqdm import tqdm

epoch = 10**2
lr = 1e-4
optim = torch.optim.AdamW(opt_lang_model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

k = 1000

from torch.amp import autocast, GradScaler

scaler = GradScaler()

for i in range(epoch):
    pbar = tqdm(dataloader)
    opt_lang_model.train()
    loss_epoch = []
    j = 1
    for token_ids, tgt_ids in pbar:
        optim.zero_grad()
        token_ids = token_ids.to(device)
        tgt_ids = tgt_ids.to(device)
        
        with autocast(device_type="cuda:0", dtype=torch.float16):
            outputs = opt_lang_model(
                token_seq=token_ids,
                causal_mask=torch.triu(torch.ones(max_seq_len-1, max_seq_len-1), diagonal=1).bool().to(device)
            )
            loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                tgt_ids.reshape(-1)
            )
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        loss_epoch.append(loss.item())
        
        pbar.set_description(f"Epoch {i+1}/{epoch} Loss {loss.item()*scale:.4e}")

        if j % k == 0:
            torch.save(opt_lang_model,'./models/model'+str(i)+'.pth')
            torch.save(torch.tensor(loss_epoch), './loss_values/loss_values_'+str(i)+'.pth')
            j = 1
        else:
            j += 1
    opt_lang_model.eval()
    torch.save(opt_lang_model,'./models/model'+str(i)+'.pth')
    torch.save(torch.tensor(loss_epoch), './loss_values/loss_values_'+str(i)+'.pth')