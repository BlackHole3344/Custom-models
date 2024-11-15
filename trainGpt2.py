from dataclasses import dataclass 
import torch 
import torch.nn as nn 
from torch.nn import functional as F
import math 



class CausalSelfAttention(nn.Module) : 
    
    def __init__(self , config ) : 
        super().__init__() 
        assert config.n_embd % config.n_head == 0 
        
        
        #A single big matrix of shape (768 , 3*768) for producing q , k , v . where the matrix splited into 3 vectors
        self.c_attn = nn.Linear(config.n_embd , 3 * config.n_embd) 
        
        #operator for adding more complexity on produced vectors after attention 
        self.c_proj = nn.Linear(config.n_embd , config.n_embd) 
        
        self.n_head = config.n_head 
        self.n_embd = config.n_embd 
        
        
        #Creates a causal masked using tril (lower traingular matrix) , and saving it has a buffer (not trainable param)
        self.register_buffer("bias" , torch.tril(torch.ones(config.block_size , config.block_size)).view(1 , 1 ,config.block_size , config.block_size))
        
    def forward(self , x ) : 
        B , T , C = x.size() # B : batch_size , T : seq_len , C : shape of Hidden_size 
        
        qkv = self.c_attn(x) # producing matrix 
        q , k , v = qkv.split(self.n_embd , dim = 2 ) # spliting 2304 into 3 
        
        k = k.view(B , T , self.n_head , C // self.n_head).transpose(1 , 2) # shape (batch , no_heads , seq_len , hidden_size / no_heads)
        q = q.view(B , T , self.n_head , C // self.n_head).transpose(1 , 2 ) 
        v = v.view(B , T , self.n_head , C // self.n_head).transpose(1 , 2 ) 
        
        att_s = (q @ k.transpose(-2 , -1)) * (1.0  / (math.sqrt(k.size(-1)))) 
        att_s = att_s.masked_fill(self.bias[: , : , :T , :T] == 0 , float('-inf')) 
        att_s = F.softmax(att_s , dim = -1) 
        
        y = att_s @ v 
        
        y = y.transpose(1 , 2 ).contiguous().view(B , T , C) # takes output to original dimesn (B , T , 768) 
        
        y = self.c_proj(y) 
         
        return y  
          
           


class MLP(nn.Module) : 
    
    # a feed-forward layer  
    def __init__(self,config) :
        super().__init__() 
        self.c_fc = nn.Linear(config.n_embd , 4 * config.n_embd) 
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd , config.n_embd) 
    def forward(self , x ) : 
        x = self.c_fc(x) 
        x = self.gelu(x)
        return self.c_proj(x)    
        
    
class Block(nn.Module) : 
    
    # The flow inside the Block 
    def __init__(self , config) : 
        super().__init__()
        self.ln_1  = nn.LayerNorm(config.n_embd) 
        self.attn = CausalSelfAttention(config) 
        self.ln_2 = nn.LayerNorm(config.n_embd) 
        self.mlp = MLP(config)
    
    def forward(self, x) : 
        # residual connection addition 
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x)) 
        return x      


@dataclass 
class GptConfig :    
    # configuration of model's hyper params 
    block_size : int = 1024
    vocab_size : int = 50257
    n_layer : int = 12 
    n_head : int = 12 
    n_embd : int = 768 
    
class Gpt(nn.Module) : 
    
    def __init__(self , config) : 
        super().__init__() 
        
        self.config = config 
        
        
        # a container for keeping the model components 
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size , self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size , self.config.n_embd),
            # a nested container for maintaing multiple layers of block 
            h =  nn.ModuleList([Block(config) for _ in range(config.n_layer)]) , 
            ln_f = nn.LayerNorm(config.n_embd)
             
        ))   
        # a projector to logits 
        self.lm_head = nn.Linear(config.n_embd , config.vocab_size , bias = False) 
    
    def forward(self , idx ) : # idx input token seq
        
        B ,T  = idx.size() # takes the Batch_size , Seq_len 
        # Seq_len must be less than the max block size (token window)
        assert T <= self.config.block_size , f"Cannot forward seq of length {T} , excedding block_size {self.config.block_size}"
        # creates a position mask
        pos = torch.arange(0 , T , dtype = torch.long , device = idx.device) 
        pos_emb = self.transformer.wpe(pos) # position embedding for shape (T , C)
        tok_emb = self.transformer.wte(idx) # token embedding for shape (B , T , C) 
        x = tok_emb + pos_emb  #add token_emb and pos_emb 
        for block in self.transformer.h : 
            x = block(x) 
        x = self.transformer.ln_f(x) 
        logits = self.lm_head(x) 
        return logits 
        
    # a method for loading hugging face weights and replacing them into custom model weights     
    @classmethod 
    def from_pretrained(cls , model_type) : 
        
        assert model_type in {"gpt2" , "gpt2-medium" , "gpt2-large" , "gpt2-xl"} 
        
        from transformers import GPT2LMHeadModel 
        
        print("loading weights from pretrained gpt : %s" % model_type) 
        
        # n_layers , n_heads , n_embd determined from the model_type 
        
        config_args = {
            "gpt2" : dict(n_layer = 12 , n_head = 12  , n_embd = 768)
        }[model_type]
        
        config_args["vocab_size"] = 50257 
        config_args["block_size"] = 1024 
        
        config = GptConfig(**config_args)
        model = Gpt(config) 
        sd = model.state_dict() 
        sd_keys = sd.keys() 
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")] #discard the atten mask matrix / buffer 
        
        #initing a hugging face transformer model 
        
        model_hf = GPT2LMHeadModel.from_pretrained(model_type) 
        sd_hf = model_hf.state_dict() 
        
        sd_keys_hf = sd_hf.keys() 
        
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")] 
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")] 
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        assert len(sd_keys) == len(sd_keys_hf), f"mismatched keys : {len(sd_keys_hf)} != {len(sd_keys)}"
        
        
        for s in sd_keys_hf : 
            if any(s.endswith(w) for w in transposed ):
                
                assert sd_hf[s].shape[::-1] == sd[s].shape 
                
                with torch.no_grad() : 
                    sd[s].copy_(sd_hf[s].t())
                    print("copied")
            else:
                # vanilla copy over the other parameters
                assert sd_hf[s].shape == sd[s].shape 
                with torch.no_grad():
                    sd[s].copy_(sd_hf[s])
                    print("copied")         
                
        return model 
    
    
    
    
num_return_sequences = 5 
max_length  = 30 

model = Gpt.from_pretrained("gpt2") 
model.eval() 
model.to("cuda") 



import tiktoken  

enc = tiktoken.get_encoding("gpt2") 

tokens = enc.encode("hello , python is a good programming language") 
print(tokens)
tokens = torch.tensor(tokens , dtype= torch.long) 
tokens = tokens.unsqueeze(0).repeat(num_return_sequences , 1) 
x = tokens.to("cuda")

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length : 
    with torch.no_grad() : 
        logits = model(x) # (B , T , VOCABSIZE)
        logits = logits[: , -1 , : ]
        
        probs = F.softmax(logits ,dim =-1) 
        
        topk_prob ,  topk_indices = torch.topk( probs , 50 , dim = -1 ) 
        ix = torch.multinomial(topk_prob ,1 )
        
        xcol = torch.gather(topk_indices , -1 , ix)
        
        x = torch.cat((x , xcol) , dim = 1 )
         
for i in range(num_return_sequences) :
    
    tokens = x[i , :max_length].tolist() 
    decoded = enc.decode(tokens) 
    print(">" ,decoded) 
    
    
     
            
        