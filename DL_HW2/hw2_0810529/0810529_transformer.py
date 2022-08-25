"""
Code for the in-class competetion of transformer
"""
#%%
# =============================================================================
#     load data
# =============================================================================
import os
print (os.getcwd())  # 查看目前終端機位置
import pandas as pd
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = pd.read_csv('news_data/train.csv', index_col=0)
test_data = pd.read_csv('news_data/test.csv', index_col=0)
train_data['Description'] = train_data['Title'] + ' ' + train_data['Description'] # 把title和description合在一起
test_data['Description'] = test_data['Title'] + ' ' + test_data['Description']  
train_data = train_data.drop(columns=['Title'])
train_data.head()

categories = {1:0, 2:1, 3:2, 4:3} # 把categories index從1234改為0123

train_data['Category'] = train_data['Category'].map(categories)
#train_data = train_data.drop(columns=['Class Index'])
#train_data.head()
train_cat_list = train_data['Category'].tolist()
#print(train_cat_list)

# =============================================================================
#     text preprocess
# =============================================================================
import spacy # 使用spacy作為tokenizer
print('Data preprocessing now...')
MAX_SEQ = 0
max_seq = MAX_SEQ # 可以自行設定，默認0
nlp = spacy.load("en_core_web_sm")

def tokenizer(text):
    text_doc = nlp(text)
    text_tok = [word.lower_ for word in text_doc if not word.is_punct] 
    # token完變小寫，若是標點符號就不取用
    return text_tok
train_tok = [tokenizer(text) for text in train_data['Description']]
test_tok = [tokenizer(text) for text in test_data['Description']]
# list中有各個句子的list，list裡已token

from collections import Counter

counter = Counter()
for text_tok in train_tok:
    counter.update(text_tok) # 從token找新單字加到counter
    if MAX_SEQ == 0:
        if len(text_tok) > max_seq:
            max_seq = len(text_tok) # 找最長句子

for text_tok in test_tok:
    counter.update(text_tok)

specials = ["<pad>", "<unk>"]

from torchtext.vocab import Vocab
# glove.6B.100d大約1G
vocab = Vocab(counter, min_freq=1, vectors='glove.6B.200d', specials=specials)
pad_idx = vocab["<pad>"]
embedding = vocab.vectors

def text_pipeline(text_tok, max_seq):
    text_len = len(text_tok)
    if max_seq > text_len:
        # pad text seq with <pad>
        # text2 = ['<sos>'] + text_tok + ['<pad>'] * (max_seq - text_len - 1)
        text2 = text_tok + ['<pad>'] * (max_seq - text_len)
    else:
        text2 = text_tok[:max_seq]
        text_len = len(text2)
    return [vocab[token] for token in text2], text_len
train_list = [text_pipeline(text_tok, max_seq) for text_tok in train_tok]

# =============================================================================
#     model
# =============================================================================
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerNet(nn.Module):
    def __init__(self, embed_pretrain, padding_idx, n_hid, n_class, n_head=6, n_layers=2, dropout=0.5):
        """
        n_tokens: vocab size
        embed_dim: size of vector for each token
        encoder: embedding matrix with size (n_tokens x embed_dim), can be imported from vocab

        n_class: number of classes to output
        n_head: number of attention heads for trans_encode
        n_hid: number of hidden nodes in NN part of trans_encode
        n_layers: number of trans_encoderlayer in trans_encode
        """
        super(TransformerNet, self).__init__()
        self.encoder = nn.Embedding.from_pretrained(embed_pretrain).requires_grad_(True)
        self.embed_dim = embed_pretrain.shape[1]
        self.n_tokens = embed_pretrain.shape[0]
        self.pad_idx = padding_idx

        self.pos_enc = PositionalEncoding(self.embed_dim, dropout)

        encoder_layers = TransformerEncoderLayer(self.embed_dim, n_head, n_hid, dropout)
        self.trans_enc = TransformerEncoder(encoder_layers, n_layers)

        self.fc1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(self.embed_dim, self.embed_dim//4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim//4, n_class),
            )

    def forward(self, x):                                   # input: (batch, seq)
        km = self.get_padding_mask(x)

        x = torch.transpose(x, 0, 1)                        # (seq, batch)
        x = self.encoder(x) * math.sqrt(self.embed_dim)     # (seq, batch, emb_dim)
        x = self.pos_enc(x)                                 # (seq, batch, emb_dim)

        x = self.trans_enc(x, src_key_padding_mask=km)
        x = x.mean(dim=0)                                   # (batch, emb_dim)
        x = self.fc1(x)                                     # (batch, n_class)

        return x

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def get_padding_mask(self, text):
        mask = (text == self.pad_idx).to(device)
        return mask

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + math.cos(math.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
    

# =============================================================================
#     Training part
# =============================================================================

def train():
# =============================================================================
#     train dataset
# =============================================================================
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset
    train_batch_size = 200
    class TrainDataset(Dataset):
        
        def __init__(self,train_cat_list, train_list):
            self.label = train_cat_list
            #self.title = train_data['Title']
            self.text = train_list

        def __len__(self):
            return len(self.text)
        
        def __getitem__(self, idx):
            text, text_len = self.text[idx]
            return self.label[idx], text, text_len
    train_dataset = TrainDataset(train_cat_list, train_list)
    
    def collate_train(batch):
        label_list, text_list, len_list = [], [], []
        for (label, text, text_len) in batch:
            len_list.append(text_len)
            label_list.append(label)
            text = torch.tensor(text, dtype=torch.int64)
            text_list.append(text)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.stack(text_list)
        return label_list.to(device), text_list.to(device), len_list
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_train)

# =============================================================================
#     training
# =============================================================================
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    from torch.nn.utils import clip_grad_norm_
    import torch.optim as optim
    print('Training now...')
    NUM_HID = 100   # 
    NUM_HEAD = 10   # 要能整除embed_dim(100) 因為embedding layer是glove100
    NUM_LAYERS = 2  #! 2~3, over 4 will crash
    DROPOUT = 0.5   #! 

    EPOCHS = 250
    LR = 1e-4
    CLIP_GRAD = 1

    model = TransformerNet(
        embedding,
        padding_idx = pad_idx,
        n_hid = NUM_HID,
        n_class = 4,
        n_head = NUM_HEAD,
        n_layers = NUM_LAYERS,
        dropout = DROPOUT
        )
    model.apply(init_weights)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss() #  論文attention is all u need參考
    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=1e-6,
                            betas=(0.9, 0.98))
    scheduler = CosineWarmupScheduler(optimizer=optimizer,
                                        warmup=10,
                                        max_iters=EPOCHS)

    train_loss_hist, train_acc_hist = [], []
    from tqdm import tqdm
    t = tqdm(range(EPOCHS), ncols=200, bar_format='{l_bar}{bar:15}{r_bar}{bar:-10b}', unit='epoch')
    model.train()
    for epoch in t:
        train_loss, train_acc, train_count = 0, 0, 0
        batch_acc, batch_count = 0, 0
        for batch_id, (label, text, seq_len) in enumerate(train_dataloader):
            optimizer.zero_grad()

            out = model(text)
            loss = criterion(out, label)
            loss.backward()
            clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

            batch_acc = (out.argmax(1) == label).sum().item()
            batch_count = label.size(0)

            train_loss += loss.item()
            train_acc += batch_acc
            train_count += batch_count

        scheduler.step()

        train_loss = train_loss/train_count
        train_acc = train_acc/train_count*100

        tl_post = "%2.5f" % (train_loss)
        ta_post = "%3.3f" % (train_acc)
        t.set_postfix({"T_Loss": tl_post, "T_Acc": ta_post})
        t.update(0)

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

    # plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(train_acc_hist, label="Train")
    plt.title("Average Accuracy History")
    plt.legend()
    plt.xlabel("Epochs")
    plt.show()

    plt.figure()
    plt.plot(train_loss_hist, label="Train")
    plt.title("Average Loss History")
    plt.legend()
    plt.xlabel("Epochs")
    plt.show()
    torch.save(model, 'transformer_weight.pth')


# =============================================================================
#     test and evaluate
# =============================================================================

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import math
#train() # if needed train
model = torch.load('transformer_weight.pth')
print('Loading pretrained model and weight...')
test_batch_size = 10
class TestDataset(Dataset):
    def __init__(self, test_list):
        self.text = test_list
    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        return self.text[i]
    
def collate_test(batch):
    text_list, len_list = [], []
    for (text, text_len) in batch:
        len_list.append(text_len)
        text = torch.tensor(text, dtype=torch.int64)
        text_list.append(text)
    text_list = torch.stack(text_list)
    return text_list.to(device), len_list

def text_pipeline(text_tok, max_seq):
    text_len = len(text_tok)
    if max_seq > text_len:
        text2 = text_tok + ['<pad>'] * (max_seq - text_len)
    else:
        text2 = text_tok[:max_seq]
        text_len = len(text2)
    return [vocab[token] for token in text2], text_len
print('Preparing test data... ')
test_list = [text_pipeline(text_tok, max_seq) for text_tok in test_tok]
test_dataset = TestDataset(test_list)
testloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_test)
answer_list = ['Category']
model.eval()
with torch.no_grad():
    for batch_id, (text, seq_len) in enumerate(testloader):
        out = model(text) # 從輸出找最大並變成list加到list
        out = (torch.argmax(out, dim=1)+1).tolist() # categories是1234要加一
        answer_list.extend(out)  # 使用extend加list
answer_id = ['Id']
for i in range(len(answer_list)):
    answer_id.append(i+1)
# =============================================================================
#     save as a csv file
# =============================================================================
import csv
with open("news_data\ 0810529_submission.csv", "w", newline="") as f:
    w = csv.writer(f)
    for i in range(len(answer_list)): # 寫成column形式
        w. writerow([answer_id[i],answer_list[i]])
print('Finishing submission! ')





# %%
