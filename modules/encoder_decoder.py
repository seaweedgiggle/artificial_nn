import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.att_model import pack_wrapper, AttModel
from modules.gcn import GCNFeatureExtractor

from transformers import OPTForCausalLM
from transformers import OPTModel, OPTConfig

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, GCNFeatureExtractor, classifier):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.GCNFeatureExtractor = GCNFeatureExtractor
        self.classifier = classifier

    # (bs, 49*n, 512) src
    # (bs, 49*n, 512) encoder
    # (bs, 14, 512) gcn_encoded -> decoder k, v
    # (bs, 49*n, 512) decoded (bs, 1, 512) -> (bs, 2, 512) ... -> (bs, max_len, 512)
    # (bs, 49)  (bs, 512) -> (bs, vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        gcn_feats = self.gcn_encode(src, src_mask)
        return self.decode(gcn_feats, src_mask, tgt, tgt_mask), self.classifier(gcn_feats[:, 1: , :])
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def gcn_encode(self, x, mask):
        x = self.encode(x, mask) 
        return self.GCNFeatureExtractor(x)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask):
    #     return self.decoder(self.tgt_embed(tgt), hidden_states, None, tgt_mask)
    
    # ?????? OPT ??? forward ??????
    # ??? encoder ????????? decoder ???????????????
#         print("hidden_states: " + str(hidden_states.size()))
#         print("target_embedding: " + str(self.tgt_embed(tgt).size()))
#         print(src_mask.shape)
#         print("target_mask: " + str(tgt_mask.size()))
        
        tgt_embedding = self.tgt_embed(tgt)
        inputs = torch.cat((hidden_states, tgt_embedding), dim=1)
        if tgt_mask.shape[0] == 1:
            mask = torch.ones((tgt_embedding.shape[0], tgt_embedding.shape[1] + hidden_states.shape[1])).cuda()
        else:
            mask1 = torch.ones((tgt_mask.shape[0], hidden_states.shape[1])).cuda()
            mask = torch.cat((mask1, tgt_mask), dim=-1)
        
        logits = self.decoder(inputs_embeds = inputs,
                    attention_mask = mask, return_dict = False)[0]
#         print("logits:" + str(logits.shape))
        return logits


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, x, mask):
        #x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        #return self.sublayer[1](x, self.feed_forward)
        residual = x
        x = self.self_attn(x, x, x, mask)
        x += residual
        x = self.norm1(x)
        residual = x
        x = self.feed_forward(x)
        x += residual
        x = self.norm2(x)
        return x



class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, x, hidden_states, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(d_model, eps = 1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps = 1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps = 1e-6)

    def forward(self, x, hidden_states, src_mask, tgt_mask):
        m = hidden_states
        residual = x
        x = self.self_attn(x, x, x, tgt_mask)
        x += residual
        x = self.norm1(x)
        residual = x
        x = self.src_attn(x, m, m, src_mask)
        x += residual
        x = self.norm2(x)
        residual = x
        x = self.feed_forward(x)
        x += residual
        x = self.norm2(x)
        return x



class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# class Embeddings(nn.Module):
#     def __init__(self, d_model, vocab):
#         super(Embeddings, self).__init__()
#         self.lut = nn.Embedding(vocab, d_model)
#         self.d_model = d_model

#     def forward(self, x):
#         return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ?????????
# mode = "global": ???????????????????????? -> ????????? (bs, 1, c)
# mode = "local": ???????????????????????? -> ????????? (bs, 20, c)
# ?????????(bs, 20, 4)
# ??????????????? local
# class classifier(nn.Module):
#     def __init__(self, mode):
#         super(classifier, self).__init__()
#         self.input_dim = 512
#         self.hidden_dim = 128
#         self.output_dim = 4
#         # TODO
#         if mode == "global":
#             self.linear1 = nn.Linear(1 * self.input_dim, 20 * self.hidden_dim)
#         elif mode == "local":
#             self.linear1 = nn.Linear(20 * self.input_dim, 20 * self.hidden_dim)
            
#         self.linear2 = nn.Linear(20 * self.hidden_dim, 20 * self.output_dim)
    
#     def forward(self, x):
# #         print(x.shape)
#         x = x.contiguous().view(x.shape[0], -1)
        
#         return self.linear2(F.relu(self.linear1(x))).view(x.shape[0], -1, self.output_dim);

# class classifier(nn.Module):
#     def __init__(self, mode):
#         super(classifier, self).__init__()
#         self.input_dim = 512
#         self.hidden_dim = 128
#         self.output_dim = 4
#         self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(self.hidden_dim, self.output_dim)
    
#     def forward(self, x):
        
#         return self.linear2(self.relu(self.linear1(x)))
    
class classifier(nn.Module):
    def __init__(self, mode):
        super(classifier, self).__init__()
        self.input_dim = 512
        self.hidden_dim = 128
        self.output_dim = 4
        self.cls = nn.ModuleList([nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        ) for _ in range(20)])
    
    def forward(self, x):
        # x (bs, 20, 512)
        prob = [self.cls[i](x[:, i].unsqueeze(1)) for i in range(20)]
        return torch.cat(prob, dim = 1)
            

class EncoderDecoder(AttModel):

    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
#         position = PositionalEncoding(self.d_model, self.dropout)
        gcnfe = GCNFeatureExtractor(self.num_classes, self.fw_adj, self.bw_adj)
        
        clr = classifier("local")
        
        # OPT
#         configuration = OPTConfig(num_attention_heads=16, vocab_size=tgt_vocab, hidden_size=1024)
#         decoder = OPTForCausalLM(configuration).from_pretrained("facebook/opt-350m")
        decoder = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
#             Decoder(
#                 DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout),
#                 self.num_layers),
            decoder,
            lambda x: x,
#             nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            decoder.model.decoder.embed_tokens,
            gcnfe,
            clr
            )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.fw_adj = args.fw_adj
        self.bw_adj = args.bw_adj
        self.num_classes = args.num_classes

        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
#         memory = self.model.encode(att_feats, att_masks)
        memory = self.model.gcn_encode(att_feats, att_masks)
        # ?????? 20 ??????????????? ???????????????
        cls_prob = self.model.classifier(memory[:, 1:, :])
        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks, cls_prob

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

#             seq_mask = seq_mask.unsqueeze(-2)
#             seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None
        return att_feats, seq, att_masks, seq_mask

    def check_mask(self, att_masks):
        sum = 1
        for i in att_masks.shape:
            sum *= i
        return bool(att_masks.sum() == sum)

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        
        # ????????? Transformer ?????? forward
        out, classify_outputs = self.model(att_feats, seq, att_masks, seq_mask)

#         outputs = F.log_softmax(self.logit(out), dim=-1)
        # OPT ????????? out ????????? Logit ???
        outputs = F.log_softmax(out, dim=-1)

#         classify_outputs = F.softmax(classify_out, -1)yy
        return outputs, classify_outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]
