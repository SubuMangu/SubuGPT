import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import json
import os
import urllib
import numpy as np
## Input output labels Dataset and Dataloader Class
class GPTDataset(Dataset):
    def __init__(self,txt,window_size,tokeniser,stride):
        # Initialise features and labels
        self.features=[]
        self.labels=[]
        # Tokenize text
        encoded_text=tokeniser.encode(txt,allowed_special={"<|endoftext|>"})
        token_size=len(encoded_text)
        # Check if there is sufficient tokens
        assert token_size>window_size,"No of tokens must be greater than window_size"
        # Generate features and labels using sliding window
        for i in range(0,token_size-window_size,stride):
            input_chunk=encoded_text[i:i+window_size]
            output_chunk=encoded_text[i+1:i+window_size+1]
            self.features.append(input_chunk)
            self.labels.append(output_chunk)
        # Convert feature and labels to torch tensor
        self.features=torch.tensor(self.features)
        self.labels=torch.tensor(self.labels)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index],self.labels[index]

def create_dataloader(txt,max_length=256,shuffle=True,num_workers=0,batch_size=4,stride=128,drop_last=True):
    # Initiate tokenizer
    bpt=tiktoken.get_encoding("gpt2")
    # Create Dataset from text
    dataset=GPTDataset(txt,max_length,bpt,stride)
    # Create Dataloader from Dataset
    dataloader=DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers
    )
    # Return Dataloader
    return dataloader

## Multihead Attention
class MultiheadAttention(nn.Module):
    def __init__(self,dim,dropout,num_heads,context_length,qkv_bias=False):
        super().__init__()
        # Check if `dim` is divisable by `num_heads`
        assert dim % num_heads==0,\
        "dim must be divisible by num_heads"
        # Declare Dimention,Number of heads and head dimention
        self.dim=dim
        self.num_heads=num_heads
        self.head_dim=dim//num_heads
        # Declare `W_q`,`W_k` and `W_v`.
        self.W_q=nn.Linear(dim,dim,bias=qkv_bias)
        self.W_k=nn.Linear(dim,dim,bias=qkv_bias)
        self.W_v=nn.Linear(dim,dim,bias=qkv_bias)
        # Declare output projection,dropout and mask
        self.out_proj=nn.Linear(dim,dim)
        self.dropout=nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length,context_length),
                       diagonal=1)
        )
    def forward(self,x):
        # Find dimentions of input
        b,num_tokens,dim=x.shape
        # Find queries,keys and values by passing through `W_q`,`W_k` and `W_v`.
        queries=self.W_q(x)
        keys=self.W_k(x)
        values=self.W_v(x)
        # Splitting dimention over heads: dim -> [num_heads,head_dim] 
        queries=queries.view(b,num_tokens,self.num_heads,self.head_dim)
        keys=keys.view(b,num_tokens,self.num_heads,self.head_dim)
        values=values.view(b,num_tokens,self.num_heads,self.head_dim)
        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim) 
        queries=queries.transpose(1,2)
        keys=keys.transpose(1,2)
        values=values.transpose(1,2)
        # Find attention scores of each head and apply mask
        attn_scores=queries@keys.transpose(2,3)
        attn_scores=attn_scores.masked_fill(self.mask[:num_tokens,:num_tokens].bool(),-torch.inf)
        # Find attention weights of each head
        attn_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1)
        attn_weights=self.dropout(attn_weights)
        # Find context vector of each head
        context_vectors=(attn_weights@values).transpose(1,2)
        # Combine context vector of each head and apply projection layer
        context_vectors=context_vectors.contiguous().view(b,num_tokens,self.dim)
        context_vectors=self.out_proj(context_vectors)
        return context_vectors
## GPT Model
class LayerNorm(nn.Module):
    def __init__(self,dim):
        # Declare eps,scale and shift
        super().__init__()
        self.eps=1e-5
        self.scale=nn.Parameter(torch.ones(dim))
        self.shift=nn.Parameter(torch.zeros(dim))
    def forward(self,x):
        # Find mean and variance
        mean=x.mean(dim=-1, keepdim=True)
        var=x.var(dim=-1, keepdim=True, unbiased=False)
        # Find z-score normalization
        # eps added to variance to prevent divided by zero error
        x=(x-mean)/torch.sqrt(var+self.eps)
        # Multiply scale and add shift,then return
        return self.scale*x+self.shift
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi))*(x + 0.044715 * torch.pow(x, 3))
        ))
class FeedForward(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(dim,4*dim),# Expanding Layer
            GELU(),# Activation Function
            nn.Linear(4*dim,dim)# Contracting Layer
        )
    def forward(self,x):
        return self.layers(x)
GPT2_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 256, # Context length
    "dim": 768,         # Embedding dimension
    "num_heads": 12,          # Number of attention heads
    "num_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        # Declare Attention,FeedForward,LayerNorm and Dropout layers
        self.attn=MultiheadAttention(
            dim=cfg["dim"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["num_heads"],
            context_length=cfg["context_length"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff=FeedForward(cfg["dim"])
        # Separate LayerNorm layers are used for each
        self.norm1=LayerNorm(cfg["dim"])
        self.norm2=LayerNorm(cfg["dim"])
        self.dropout=nn.Dropout(cfg["drop_rate"])
    def forward(self,x):
        # Passing through Attention layer
        shortcut=x
        x=self.norm1(x)
        x=self.attn(x)
        x=self.dropout(x)
        x=x+shortcut
        # Passing through Feed Feedforward layer
        shortcut=x
        x=self.norm2(x)
        x=self.ff(x)
        x=self.dropout(x)
        x=x+shortcut

        return x
class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb=nn.Embedding(cfg["vocab_size"],cfg["dim"])
        self.pos_emb=nn.Embedding(cfg["context_length"],cfg["dim"])
        self.dropout=nn.Dropout(cfg["drop_rate"])
        self.trf_blocks=nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["num_layers"])]
        )
        self.layer_norm=LayerNorm(cfg["dim"])
        self.out_head=nn.Linear(
            cfg["dim"],cfg["vocab_size"],bias=False
        )
    def forward(self,input):
        batch_size,context_len=input.shape
        tok_emb=self.tok_emb(input)
        pos_emb=self.pos_emb(torch.arange(context_len,device=input.device))
        x=tok_emb+pos_emb
        x=self.dropout(x)
        x=self.trf_blocks(x)
        x=self.layer_norm(x)
        x=self.out_head(x)
        return x
## Generating Text
def generate_text_ids(model,input,new_tokens,window_size):
    for _ in range(new_tokens):
        input_cropped=input[:,-window_size:]
        with torch.no_grad():
            logits=model(input_cropped)
        logits=logits[:,-1,:]
        probas=torch.softmax(logits,dim=-1)
        next_token_id=torch.argmax(probas,dim=-1,keepdim=True)
        input=torch.cat((input,next_token_id),dim=-1)
    return input
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def calculate_batch_loss(input_batch,target_batch,model,device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits=model(input_batch)
    cross_entropy_loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())
    return cross_entropy_loss

def calculate_dataloader_loss(dataloader: torch.utils.data.DataLoader ,model: GPTModel,device: torch.device)-> torch.Tensor:
    total_loss=0
    for input_batch,target_batch in dataloader:
        loss=calculate_batch_loss(input_batch,target_batch,model,device)
        total_loss+=loss
    avg_loss=total_loss/len(dataloader)
    return avg_loss

def train_model(model:GPTModel,train_loader:torch.utils.data.DataLoader,val_loader:torch.utils.data.DataLoader,num_epochs:int,start_text: str,tokenizer:tiktoken,new_tokens:int,learning_rate:float,device: torch.device):
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
    model.to(device)
    train_losses,val_losses,epochs=[],[],[]
    for epoch in range(num_epochs):
        model.train()# set to training mode
        for input_batch,target_batch in train_loader:
            optimizer.zero_grad()# Resetting grad to zero
            loss=calculate_batch_loss(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()# updates the weights
        model.eval()# set to evaluation mode
        with torch.no_grad():
            train_loss=calculate_dataloader_loss(train_loader,model,device)
            val_loss=calculate_dataloader_loss(val_loader,model,device)
            train_losses.append(train_loss.cpu().item())
            val_losses.append(val_loss.cpu().item())
            epochs.append(epoch+1)
            ids=tokenizer.encode(start_text,allowed_special={"<|endoftext|>"})
            ids=torch.tensor([ids]).to(device)
            output_ids=generate(
                model=model,
                idx=ids,
                max_new_tokens=new_tokens,
                context_size=GPT2_CONFIG["context_length"],
                temperature=1.2,
                eos_id=50256
            )
        print(f'Epoch {epoch+1}: Train loss {train_loss}, Val loss {val_loss}')
        print(tokenizer.decode(output_ids[0].tolist()))
    return train_losses,val_losses,epochs

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].attn.W_q.weight = assign(
            gpt.trf_blocks[b].attn.W_q.weight, q_w.T)
        gpt.trf_blocks[b].attn.W_k.weight = assign(
            gpt.trf_blocks[b].attn.W_k.weight, k_w.T)
        gpt.trf_blocks[b].attn.W_v.weight = assign(
            gpt.trf_blocks[b].attn.W_v.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].attn.W_q.bias = assign(
            gpt.trf_blocks[b].attn.W_q.bias, q_b)
        gpt.trf_blocks[b].attn.W_k.bias = assign(
            gpt.trf_blocks[b].attn.W_k.bias, k_b)
        gpt.trf_blocks[b].attn.W_v.bias = assign(
            gpt.trf_blocks[b].attn.W_v.bias, v_b)

        gpt.trf_blocks[b].attn.out_proj.weight = assign(
            gpt.trf_blocks[b].attn.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].attn.out_proj.bias = assign(
            gpt.trf_blocks[b].attn.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.layer_norm.scale = assign(gpt.layer_norm.scale, params["g"])
    gpt.layer_norm.shift = assign(gpt.layer_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)
config={
        'vocab_size': 50257,
    'context_length': 1024,
    'drop_rate': 0.0,
    'qkv_bias': True,
    'dim': 1024,
    'num_layers': 24,
    'num_heads': 16
    }

