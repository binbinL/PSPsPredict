import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import math

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.3, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        # self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, att_weight = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.leakyrelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src,att_weight

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.2, max_len=2000):
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
        x = x + self.pe[:x.size(0), :] # +torch.Size([32, 1, 256])
        return self.dropout(x)
    

class LinearDecoder(nn.Module):
    def __init__(self,d_model,n_class,dropout):
        super(LinearDecoder, self).__init__()
        self.decoder1 = nn.Linear(d_model, d_model // 2)
        self.decoder2 = nn.Linear(d_model // 2, d_model // 4)
        self.classifier = nn.Linear(d_model // 4, n_class)
        self.norm1 = nn.LayerNorm(d_model // 2)
        self.norm2 = nn.LayerNorm(d_model // 4)
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.norm1(self.dropout(self.tanh(self.decoder1(x))))
        x_mid = self.norm2(self.dropout(self.leakyrelu(self.decoder2(x))))
        x = self.dropout(self.classifier(x_mid))
        return x,x_mid
    
    
class K_mer_aggregate(nn.Module):
    def __init__(self,kmers,in_dim,out_dim,dropout=0.1):# 0.2
        '''
        x:  (batch_size, sequence_length, features)
        return:  (batch_size, sequence_length, features)
        '''
        super(K_mer_aggregate, self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.convs=[]
        for i in kmers:
            # sequence_length -> sequence_length-i+1
            # self.convs.append(nn.Conv1d(in_dim,out_dim,i,padding=0))
            # sequence_length -> sequence_length: only one convs  kmer 5 -> pad 2
            pad = (i-1)//2
            self.convs.append(nn.Conv1d(in_dim,out_dim,i,padding=pad))
        self.convs=nn.ModuleList(self.convs)
        self.activation=nn.ReLU(inplace=True)
        self.norm=nn.LayerNorm(out_dim)

    def forward(self,x):
        # 卷积是在最后一个维度上做的
        # (batch_size, sequence_length, features)->(batch_size, features, sequence_length)
        x = x.permute(0,2,1)
        outputs=[]
        for conv in self.convs:
            outputs.append(conv(x))
        # 拼接
        outputs=torch.cat(outputs,dim=2) 
        # value add
        # outputs = torch.stack(outputs, dim=2)
        # outputs = torch.sum(outputs, dim=2)
        outputs=self.norm(outputs.permute(0,2,1))

        return outputs

class MyModel(nn.Module):
    def __init__(self, d_embedding, d_model, dropout, n_class, vocab_size,nlayers,nhead,dim_feedforward,kmers):
        super(MyModel, self).__init__()
        self.embeding = nn.Embedding(vocab_size,d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.kmer_aggregation = K_mer_aggregate(kmers,d_model,d_model)

        self.transformer_encoder = []
        for i in range(nlayers):
            self.transformer_encoder.append(TransformerEncoderLayer(d_model, nhead, dim_feedforward))
        self.transformer_encoder= nn.ModuleList(self.transformer_encoder)
        self.dff = d_embedding + dim_feedforward
        self.decoder = LinearDecoder(self.dff,n_class,dropout)

    def forward(self, x1,x2): #x2 - seq
        x1 = x1.to(torch.float32)
        x2 = self.embeding(x2) # torch.Size([32, 1024])->torch.Size([32, 1024, 256])
        x2 = self.pos_encoder(x2) # torch.Size([32, 1024, 256])

        x2 = self.kmer_aggregation(x2) #torch.Size([32, 2041, 256]) exp: 1024-4+1 + 1024-5+1 pad = 0
        # print(x2.shape)
        # (batch_size, sequence_length , features)->(sequence_length, batch_size, features)
        x2 = x2.permute(1,0,2)
        # torch.Size([1024, 32, 256])

        attention_weights = []
        for layer in self.transformer_encoder:
            x2,attention_weights_layer=layer(x2)
            attention_weights.append(attention_weights_layer)
        attention_weights=torch.stack(attention_weights)

        x2 = x2.permute(1,0,2)
        x2 = torch.mean(x2, dim=2)
        x_mid = torch.cat((x1, x2), dim=1)
        x,x_mid = self.decoder(x_mid)
        return x,x_mid
    
class MyModelwoKmer(nn.Module):
    def __init__(self, d_embedding, dropout, n_class):
        super(MyModelwoKmer, self).__init__()

        self.decoder = LinearDecoder(d_embedding,n_class,dropout)

    def forward(self, x): 
        x = x.to(torch.float32)
        out,x_mid = self.decoder(x)
        return out,x_mid
    
# class MyModelwoProtT5(nn.Module):
#     def __init__(self,  d_model, dropout, n_class, vocab_size,nlayers,nhead,dim_feedforward,kmers):
#         super(MyModelwoProtT5, self).__init__()
#         self.embeding = nn.Embedding(vocab_size,d_model)
#         self.pos_encoder = PositionalEncoding(d_model)
#         self.kmer_aggregation = K_mer_aggregate(kmers,d_model,d_model)

#         self.transformer_encoder = []
#         for i in range(nlayers):
#             self.transformer_encoder.append(TransformerEncoderLayer(d_model, nhead, dim_feedforward))
#         self.transformer_encoder= nn.ModuleList(self.transformer_encoder)
#         self.decoder = LinearDecoder(dim_feedforward,n_class,dropout)

#     def forward(self, x2): 
#         x2 = self.embeding(x2) 
#         x2 = self.pos_encoder(x2) 
#         x2 = self.kmer_aggregation(x2) 
#         x2 = x2.permute(1,0,2)

#         attention_weights = []
#         for layer in self.transformer_encoder:
#             x2,attention_weights_layer=layer(x2)
#             attention_weights.append(attention_weights_layer)
#         attention_weights=torch.stack(attention_weights)

#         x2 = x2.permute(1,0,2)
#         x2 = torch.mean(x2, dim=2)
#         out,x_mid = self.decoder(x2)
#         return out,x_mid
    


class MyModel_woKmerConv(nn.Module):
    def __init__(self, d_embedding, d_model, dropout, n_class, vocab_size,nlayers,nhead,dim_feedforward):
        super(MyModel_woKmerConv, self).__init__()
        self.embeding = nn.Embedding(vocab_size,d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer_encoder = []
        for i in range(nlayers):
            self.transformer_encoder.append(TransformerEncoderLayer(d_model, nhead, dim_feedforward))
        self.transformer_encoder= nn.ModuleList(self.transformer_encoder)
        self.dff = d_embedding + dim_feedforward
        self.decoder = LinearDecoder(self.dff,n_class,dropout)

    def forward(self, x1,x2): #x2 - seq
        x1 = x1.to(torch.float32)
        x2 = self.embeding(x2) # torch.Size([32, 1024])->torch.Size([32, 1024, 256])
        x2 = self.pos_encoder(x2) # torch.Size([32, 1024, 256])
        x2 = x2.permute(1,0,2)
        # torch.Size([1024, 32, 256])

        attention_weights = []
        for layer in self.transformer_encoder:
            x2,attention_weights_layer=layer(x2)
            attention_weights.append(attention_weights_layer)
        attention_weights=torch.stack(attention_weights)

        x2 = x2.permute(1,0,2)
        x2 = torch.mean(x2, dim=2)
        x_mid = torch.cat((x1, x2), dim=1)
        x,x_mid = self.decoder(x_mid)
        return x,x_mid

class MyModel_woMH(nn.Module):
    def __init__(self, d_embedding, d_model, dropout, n_class, vocab_size,dim_feedforward,kmers):
        super(MyModel_woMH, self).__init__()
        self.embeding = nn.Embedding(vocab_size,d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.kmer_aggregation = K_mer_aggregate(kmers,d_model,d_model)

        self.dff = d_embedding + dim_feedforward
        self.decoder = LinearDecoder(self.dff,n_class,dropout)

    def forward(self, x1,x2): #x2 - seq
        x1 = x1.to(torch.float32)
        x2 = self.embeding(x2) # torch.Size([32, 1024])->torch.Size([32, 1024, 256])
        x2 = self.pos_encoder(x2) # torch.Size([32, 1024, 256])

        x2 = self.kmer_aggregation(x2) #torch.Size([32, 2041, 256]) exp: 1024-4+1 + 1024-5+1 pad = 0
        x2 = torch.mean(x2, dim=2)
        x_mid = torch.cat((x1, x2), dim=1)
        x,x_mid = self.decoder(x_mid)
        return x,x_mid