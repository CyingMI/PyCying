import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cying import nn as cynn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from torchinfo import summary
num_class = 10**4
opt_lang_model = cynn.OptLangModel(
    num_class,
    512,
    4,
    2048,
    10
).to(device)
summary(opt_lang_model)

import json
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader

scale = 8
max_seq_len = 1024
batch_size = 4

class Trans2019(Dataset):
    def __init__(self, filepath, tokenizer, seq_len=64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data_english = []
        self.data_chinese = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                datai = json.loads(line)
                self.data_english.append(datai['english'])
                self.data_chinese.append(datai['chinese'])
    
    def __getitem__(self, index):
        text_english = self.data_english[index]
        text_chinese = self.data_chinese[index]
        seq_len = len(text_english) + len(text_chinese)
        cut_rand = torch.rand(1)
        while seq_len/self.seq_len < cut_rand:
            index_rand = torch.randint(0,self.__len__(),(1,))[0]
            text_english += self.data_english[index_rand]
            text_chinese += self.data_chinese[index_rand]
            seq_len = len(text_english) + len(text_chinese)
        if torch.randn(1) < 0:
            tokens_english = [0] + self.tokenizer.encode(text_english).ids + [1]
            tokens_chinese = [2] + self.tokenizer.encode(text_chinese).ids + [3]
            token_seq = tokens_english + tokens_chinese[:-1]
            tgt_seq = [4]*len(tokens_english) + tokens_chinese[1:]
        else:
            tokens_english = [2] + self.tokenizer.encode(text_english).ids + [3]
            tokens_chinese = [0] + self.tokenizer.encode(text_chinese).ids + [1]
            token_seq = tokens_chinese + tokens_english[:-1]
            tgt_seq = [4]*len(tokens_chinese) + tokens_english[1:]
        if len(token_seq) >= self.seq_len:
            token_seq = token_seq[:self.seq_len]
            tgt_seq = tgt_seq[:self.seq_len]
        else:
            token_seq = token_seq + [4]*(self.seq_len - len(token_seq))
            tgt_seq = tgt_seq + [4]*(self.seq_len - len(tgt_seq))
        input_ids = torch.tensor(token_seq, dtype=torch.long)
        target_ids = torch.tensor(tgt_seq, dtype=torch.long)
        return input_ids, target_ids

    def __len__(self):
        return len(self.data_english) // batch_size * batch_size

tokenizer = Tokenizer.from_file("../cying/datasets/tokenizer-wiki.json")
trans_dataset = Trans2019(
    filepath=f'../cying/datasets/translation2019zh/translation2019zh_train.json',
    tokenizer=tokenizer,
    seq_len=max_seq_len
)
train_loader = DataLoader(
    dataset=trans_dataset,
    batch_size=batch_size,
    shuffle=True
)
valid_dataset = Trans2019(
    filepath=f'../cying/datasets/translation2019zh/translation2019zh_valid.json',
    tokenizer=tokenizer,
    seq_len=max_seq_len
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=batch_size
)

from tqdm import tqdm
epoch = 10**2
lr = 1e-3
optim = torch.optim.Adam(opt_lang_model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss(ignore_index=4)

for i in range(epoch):
    pbar = tqdm(train_loader)
    k = 1
    j = 1
    opt_lang_model.train()
    for token_ids, tgt_ids in pbar:
        token_ids = token_ids.to(device)
        tgt_ids = tgt_ids.to(device)
        outputs = opt_lang_model(
            token_seq=token_ids,
            causal_mask=torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool().to(device)
        )
        loss = criterion(
            outputs.view(-1, outputs.size(-1)),
            tgt_ids.reshape(-1)
        ) / scale
        loss.backward()
        if k % scale == 0:
            k = 1
            optim.step()
            optim.zero_grad()
        else:
            k += 1
        pbar.set_description(f"Epoch {i+1}/{epoch} Loss {loss.item()*scale:.4e}")
        j += 1
        if j % 1000 == 0:
            torch.save(opt_lang_model,'../models/model.pth')
    with torch.no_grad():
        opt_lang_model.eval()
        pbar = tqdm(valid_loader)
        for token_ids, tgt_ids in pbar:
            token_ids = token_ids.to(device)
            tgt_ids = tgt_ids.to(device)
            outputs = opt_lang_model(
                token_seq=token_ids,
                causal_mask=torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool().to(device)
            )
            loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                tgt_ids.reshape(-1)
            )
            pbar.set_description(f"Epoch {i+1}/{epoch} Loss {loss.item():.4e}")
    torch.save(opt_lang_model,'./models/model'+str(i)+'.pth')