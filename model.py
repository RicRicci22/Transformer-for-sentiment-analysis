import torch
import torch.nn as nn 

class SelfAttention(nn.Module):
    def __init__(self,embed_size, heads):
        super(SelfAttention,self).__init__()
        # If embed_size 256 and 8 heads, we will split in chunks of 256/8 = 32
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim*heads == embed_size), "Problem with embed size or number of heads"
        
        self.values = nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.keys = nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.queries = nn.Linear(self.head_dim,self.head_dim,bias=False)
        
        self.fcout = nn.Linear(heads*self.head_dim,embed_size)
    
    def forward(self,V,K,Q,mask):
        N = Q.shape[0]
        value_len,key_len,query_len = V.shape[1],K.shape[1],Q.shape[1]
        
        # Divide the embedding into the heads. 
        V = V.reshape(N,value_len,self.heads,self.head_dim)
        K = K.reshape(N,key_len,self.heads,self.head_dim)
        Q = Q.reshape(N,query_len,self.heads,self.head_dim)
        
        V = self.values(V)
        K = self.keys(K)
        Q = self.queries(Q)
        
        output = torch.einsum("nqhd,nkhd->nhqk",[Q,K])
        # Query shape: (N,query_len,heads,heads_dim)
        # Keys shape: (N,key_len,heads,heads_dim)
        
        if mask is not None:
            output = output.masked_fill(mask==0,float('-1e20')) # Filla output con il numero che si mette guardando la condizione su un altro tensore
        
        attention = torch.softmax(output/(self.embed_size**0.5),dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd",[attention,V]).reshape((N,query_len,self.heads*self.head_dim))
        
        out = self.fcout(out)
        
        return out
    

class TransformerBlock(nn.Module):
    def __init__(self,embed_size,heads,dropout,forward_expansion) -> None:
        # forward_expansion serve per capire la dimensione di output del primo linear layer in uscita
        super(TransformerBlock,self).__init__()
        self.attention = SelfAttention(embed_size,heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size,forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size,embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,value,key,query,mask):
        attention = self.attention(value,key,query,mask)
        x = self.dropout(self.norm1(attention+query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x))
        return out

class Encoder(nn.Module):
    def __init__(self,src_vocab_size,embed_size,n_layers,heads,device,forward_exp,dropout,max_length,modality) -> None:
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.modality = modality
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size,embed_size)
        self.pos_embedding = nn.Embedding(max_length,embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size,heads,dropout=dropout,forward_expansion=forward_exp)
            ]*n_layers
        )
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(embed_size,2) # To perform sentiment analysis
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self,x,mask):
        N,seq_len = x.shape
        positions = torch.arange(0,seq_len).expand(N,seq_len).to(self.device)
    
        out = self.word_embedding(x)+self.pos_embedding(positions)
        #out = self.dropout(out)
        
        for layer in self.layers:
            out = layer(out,out,out,mask)
        
        # To do sentiment analysis I max pool or average over the encoded embeddings
        if self.modality=='max':
            out = torch.amax(out,dim=1)
        elif self.modality=='mean':
            out = torch.mean(out, dim=1)

        out = self.linear(out)
        out = self.logsoftmax(out)
        
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self,embed_size, heads, forward_exp, dropout) -> None:
        super(DecoderBlock,self).__init__()
        self.attention = SelfAttention(embed_size,heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size,heads,dropout,forward_exp)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x,value,key,src_mask,trg_mask):
        attention = self.attention(x,x,x,trg_mask)
        query = self.dropout(self.norm(attention+x))
        out = self.transformer_block(value,key,query,src_mask)
        return out


class Decoder(nn.Module):
    def __init__(self,trg_vocab_size,embed_size,num_layers,heads,forward_exp,dropout,device,max_length) -> None:
        super(Decoder,self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size,embed_size)
        self.pos_embedding = nn.Embedding(max_length,embed_size)
        
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size,heads,forward_exp,dropout)]*num_layers
        )
        
        self.fc_out = nn.Linear(embed_size,trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x,enc_out,src_mask,trg_mask):
        N,seq_len = x.shape
        positions = torch.arange(0,seq_len).expand(N,seq_len).to(self.device)
        x = self.dropout(self.word_embedding(x)+self.pos_embedding(positions))
        
        for layer in self.layers:
            x = layer(x,enc_out,enc_out,src_mask,trg_mask)
        
        out = self.fc_out(x)
        
        return out


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size=256,
                 num_layers=6,
                 forward_expansion=4,
                 heads=8,
                 dropout=0.1,
                 device="cuda",
                 max_len=100) -> None:
        super(Transformer,self).__init__()
        
        self.encoder = Encoder(src_vocab_size,embed_size,num_layers,heads,device,forward_expansion,dropout,max_len)
        self.decoder = Decoder(trg_vocab_size,embed_size,num_layers,heads,forward_expansion,dropout,device,max_len)
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self,src):
        src_mask = (src!=self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        
        return src_mask.to(self.device)
    
    def make_trg_mask(self,trg):
        N,trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len,trg_len))).expand(N,1,trg_len,trg_len)
        
        return trg_mask.to(self.device)
    
    def forward(self,src,trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src,src_mask)
        out = self.decoder(trg,enc_src,src_mask,trg_mask)
        
        return out

class Transformer_sentiment(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 src_pad_idx,
                 embed_size=256,
                 num_layers=6,
                 forward_expansion=2,
                 heads=8,
                 dropout=0.1,
                 device="cuda",
                 max_len=100, 
                 modality='max') -> None:
        super(Transformer_sentiment,self).__init__()
        
        self.encoder = Encoder(src_vocab_size,embed_size,num_layers,heads,device,forward_expansion,dropout,max_len,modality)
        
        self.src_pad_idx = src_pad_idx
        self.device = device
    
    def make_src_mask(self,src):
        src_mask = (src!=self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        
        return src_mask.to(self.device)
    
    def forward(self,src):
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src,src_mask)
        
        return enc_src
    
    
        

# if __name__=="__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     x = torch.tensor([[1,5,6,7,8,4,5,6,3,5],[1,5,6,7,8,5,6,2,3,5]]).to(device)
#     y = torch.tensor([[1,5,6,7,8],[1,5,6,7,8]]).to(device)
    
#     src_pad_idx = 0
#     trg_pad_idx = 0
#     src_vocab_size = 10
#     trg_vocab_size = 10
    
#     model = Transformer(src_vocab_size,trg_vocab_size,src_pad_idx,trg_pad_idx).to(device)
#     #print(y[:,1:])
#     print(x.shape)
#     out = model(x,y[:,1:])
    
#     print(out.shape)

    