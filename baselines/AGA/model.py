from einops import rearrange
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


class PreNorm1D(nn.Module):
    def __init__(self, q, k, v, fn):
        super().__init__()
        self.fn = fn
        self.q_norm = RMSNorm(q)
        self.k_norm = RMSNorm(k)
        self.v_norm = RMSNorm(v)

    def forward(self, q, k=None, v=None, *args, **kwargs):
        q = self.q_norm(q)
        if k is not None:
            k = self.k_norm(k)
            v = self.v_norm(v)
            return self.fn(q, k, v, *args, **kwargs)
        else:
            return self.fn(q, *args, **kwargs)
    
        
class FeedForwardNetwork(nn.Module):
    def __init__(self, dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)


    
class Attention1D(nn.Module):
    def __init__(self, q, k, v, out, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.q_f = nn.Linear(q, out)
        self.k_f = nn.Linear(k, out)
        self.v_f = nn.Linear(v, out)
        self.to_out = nn.Sequential(
            nn.Linear(out, out),
            nn.Dropout(dropout),
        )

    def forward(self, q, k, v):
        q, k, v = self.q_f(q), self.k_f(k), self.v_f(v)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k, v])
        dots = (q @ k.transpose(-1, -2)) * (q.size(-1) ** -0.5)
        att_w = dots.softmax(-1)
        out = att_w @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), att_w.mean(1)
    

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Retrieve(nn.Module):
    def __init__(self, q, k, v, out, heads=8, dropout=0):
        super().__init__()
        self.target_f1 = PreNorm1D(q, k, v, Attention1D(q, k, v, out, heads=heads, dropout=dropout))
        self.target_f2 = Residual(PreNorm1D(out, out, out, FeedForwardNetwork(out, dropout=dropout)))
        
    def forward(self, q, k, v):
        out, att_w = self.target_f1(q, k, v)
        return self.target_f2(out), att_w
        
    
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        if x.ndim == 2:
            g = self.g[None, :]
        else:
            g = self.g[None, None, :]
        return g * (x / ((x ** 2).mean(-1, keepdim=True) ** 0.5))
    
    
class ScaleNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        if x.ndim == 2:
            g = self.g[None, :]
        else:
            g = self.g[None, None, :]
        return g * F.normalize(x, p=2, dim=-1)
    
    
class QueueMemory(nn.Module):
    def __init__(self, max_length, hidden_dim, *args, **kwargs):
        super().__init__()
        self.key_queue = []
        self.value_queue = []
        self.max_length = max_length
        
    def empty(self):
        return len(self.key_queue) == 0
    
    def reset(self, batch_size):
        self.key_queue = []
        self.value_queue = []
    
    def read(self):
        key = torch.cat(self.key_queue, dim=1)
        value = torch.cat(self.value_queue, dim=1)
        return key, value
    
    def write(self, key, value):
        self.key_queue.append(key)
        self.value_queue.append(value)
        
        self.key_queue = self.key_queue[-self.max_length:]
        self.value_queue = self.value_queue[-self.max_length:]

    
class AGA(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, order=30, 
                 dropout=0.6, attention_dropout=0.6, return_embedding=False, input_norm=True, 
                 random_inductive_perturb=0, perturb_systematic_start_index=-1, recurrent_query='ma', recurrent_h=False, iam=True,
                 gate_x=True, ma_ratio=0.8, key_prob=False, query_prob=True, down_ratio=4, single_value_gate=False, top1_only=False):
        super().__init__()

        self.evidence_ratio = []
        
        # Initial configuration
        self.gate_x = gate_x
        self.iam = iam
        self.return_embedding = return_embedding
        self.random_inductive_perturb = random_inductive_perturb
        self.perturb_systematic_start_index = perturb_systematic_start_index
        self.recurrent_query = recurrent_query
        self.recurrent_h = recurrent_h
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.ma_ratio = ma_ratio
        self.key_prob = key_prob
        self.query_prob = query_prob
        self.top1_only = top1_only
        
        # Encoder module
        self.encoder = nn.Sequential(
            nn.LayerNorm(in_dim) if input_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            ScaleNorm(hidden_dim)
        )
        
        # Selector module
        self.selector = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim // down_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // down_ratio, 1 if single_value_gate else hidden_dim),
            nn.Sigmoid()
        )
        
        # Anticipation classifier
        self.anticipation_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        # Recurrent query aggregator
        if recurrent_query == 'lstm':
            self.rnn = nn.LSTMCell(out_dim, out_dim)
    
        
        # Encoding modules
        self.encode_q = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(out_dim, hidden_dim // down_ratio)
        ) if iam else nn.Linear(hidden_dim, hidden_dim // down_ratio)
        
        self.encode_k = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(out_dim, hidden_dim // down_ratio)
        )

        # Memory-related modules
        self.hx_norm = ScaleNorm(hidden_dim)
        self.memory = QueueMemory(order, hidden_dim)

        # State transform module
        self.state_transform = Retrieve(hidden_dim // down_ratio, hidden_dim // down_ratio if iam else hidden_dim, hidden_dim, hidden_dim, 
                                        dropout=attention_dropout, heads=16)

    
    def att_forward(self, q, k, v, x):
        hx, _ = self.state_transform(
            self.encode_q(q.softmax(-1)),
            self.encode_k(k), 
            v
        )
        hx = self.hx_norm(hx)

        ratio = self.selector(torch.cat([x, hx], dim=-1))
        hx = ratio * hx + (1-ratio) * x
        action = self.anticipation_classifier(hx) 
        return action


    def _leave_top1only(self, x):
        _, idx = x.topk(1, -1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, idx, 1)
        return mask
        

    def grab_state(self, x, query, gt, update_memory, perturb_this_frame=False):
        if self.memory.empty():
            hx = self.hx_norm(x)
        else:
            k, v = self.memory.read()
            
            self.att_k = k
            self.att_v = v
            self.att_x = x
            self.att_q = query

            if self.top1_only:
                k = self._leave_top1only(k)
                query = self._leave_top1only(query)

            
            k = self.encode_k(k)
            q = self.encode_q(query if self.iam else x)

            # Transform and normalize hidden state
            hx, _ = self.state_transform(
                q,
                k if self.iam else v,
                v
            )
            hx = hx.mean(1).unsqueeze(1)
            hx = self.hx_norm(hx)

            # Apply gating mechanism
            if self.gate_x:
                ratio = self.selector(torch.cat([x, hx], dim=-1))
                hx = ratio * hx + (1 - ratio) * x
                self.evidence_ratio.append(ratio)
            else:
                hx = hx + x
        

        # Generate action prediction
        action = self.anticipation_classifier(hx)
        self.final_action = action
        if perturb_this_frame:
            action = torch.randn_like(action)

        # Update memory
        if update_memory:
            action_key = action#.detach()
            if gt is not None:
                action_key = gt
            
            if self.key_prob:
                action_key = action_key.softmax(-1)
            self.memory.write(value=hx if self.recurrent_h else x, key=action_key)

        return action, hx
    
    def forward(self, seq, gt=None, update_memory=True):
        self.memory.reset(seq.size(0))
        seq_out = []
        emb_out = []
        self.evidence_ratio = []

        if gt is not None:
            gt = F.one_hot(gt+1, self.out_dim+1)[..., 1:].float()

        
        if self.recurrent_query == 'lstm':
            hx, cx = torch.zeros(seq.size(0), self.out_dim, device=seq.device, dtype=seq.dtype), torch.zeros(seq.size(0), self.out_dim, device=seq.device, dtype=seq.dtype)

        # Random perturbation index
        if self.random_inductive_perturb > 0:
            if self.perturb_systematic_start_index < 0:
                perturb_index = torch.randperm(seq.size(1) - 1)[:self.random_inductive_perturb]
            else:
                perturb_index = torch.arange(seq.size(1) - 1)[self.perturb_systematic_start_index:self.perturb_systematic_start_index+self.random_inductive_perturb]
        else:
            perturb_index = []

        # Process sequence
        for i in range(seq.size(1)):
            x = self.encoder(seq[:, i]).unsqueeze(1)
            if i > 0:
                if self.query_prob:
                    query_t = query_t.softmax(-1)
                if self.recurrent_query == 'lstm':
                    hx, cx = self.rnn(query_t.squeeze(1), (hx, cx))
                    q = hx.unsqueeze(1)
                elif self.recurrent_query == 'ma':
                    if self.query_prob:
                        q = q.softmax(-1) * (1-self.ma_ratio) + query_t * self.ma_ratio
                    else:
                        q = q * (1-self.ma_ratio) + query_t * self.ma_ratio
                else:
                    q = query_t
            else:
                q = torch.zeros(x.size(0), 1, self.out_dim, device=x.device, dtype=x.dtype)

            query_t, emb = self.grab_state(x, q, gt=gt[:, i].unsqueeze(1) if gt is not None else None, update_memory=update_memory, perturb_this_frame=i in perturb_index)
                
            seq_out.append(query_t)

            if gt is not None:
                query_t = gt[:, i].unsqueeze(1)

            if self.return_embedding:
                emb_out.append(emb)

        return (torch.cat(seq_out, 1), torch.cat(emb_out, 1)) if self.return_embedding else torch.cat(seq_out, 1)