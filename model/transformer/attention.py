import numpy as np
import torch
from torch import nn
from model.containers import Module
from torch.nn import functional as F
from ..Dy_GAT.GAT import *
from model.transformer.utils import *


class DynamicWeightGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DynamicWeightGenerator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, d_k, d_v, h):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, input_gl=None, memory=None,
                isencoder=None):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        """
        # att[0][0].argmax()
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k) 10 8 50 64
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) 10 8 64 50
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v) 10 8 50 64

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk) 10 8 50 50

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask.bool(), -np.inf)  # 10, 8, 50, 50
        att = torch.softmax(att, -1)  # 10, 8, 50, 50
        # np.save("att",att.cpu())
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
                                                                         self.h * self.d_v)  # (b_s, nq, h*d_v) 10 50 512

        out = self.fc_o(out)  # (b_s, nq, d_model) 10 50 512
        return out


class PriorKnowledgeAugmentedAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, m):
        super(PriorKnowledgeAugmentedAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.m_k = nn.Parameter(torch.FloatTensor(1, m, h * d_k))
        self.m_v = nn.Parameter(torch.FloatTensor(1, m, h * d_v))
        self.fc_mm = nn.Linear(1, h * d_v)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.m = m

        self.dynamic_weight_generator_q = DynamicWeightGenerator(d_model, d_model // 2, d_model)
        self.dynamic_weight_generator_k = DynamicWeightGenerator(d_model, d_model // 2, d_model)
        self.dynamic_weight_generator_v = DynamicWeightGenerator(d_model, d_model // 2, d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.normal_(self.m_k, 0, 1 / self.d_k)
        nn.init.normal_(self.m_v, 0, 1 / self.m)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)
        nn.init.xavier_uniform_(self.fc_mm.weight)
        nn.init.constant_(self.fc_mm.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, input_gl=None, memory=None,
                isencoder=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        # 动态权重生成和应用
        dynamic_weights_q = self.dynamic_weight_generator_q(queries)
        dynamic_weights_k = self.dynamic_weight_generator_k(keys)
        dynamic_weights_v = self.dynamic_weight_generator_v(values)

        queries = queries * dynamic_weights_q
        keys = keys * dynamic_weights_k
        values = values * dynamic_weights_v

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        memory = memory.unsqueeze(-1)
        mm = self.fc_mm(memory).view(b_s, nk, self.h, self.d_k).permute(0, 2, 1, 3)
        att = torch.matmul(q, k * (mm.permute(0, 1, 3, 2))) / np.sqrt(self.d_k)

        if attention_weights is not None:
            att = torch.cat([att[:, :, :, :nk] * attention_weights, att[:, :, :, nk:]], -1)
        if attention_mask is not None:
            att[:, :, :, :nk] = att[:, :, :, :nk].masked_fill(attention_mask, -np.inf)

        att = torch.softmax(att, -1)

        out = torch.matmul(att, v * mm).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out


class MultiHeadAttention(Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, isenc=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        if attention_module is not None:
            if attention_module_kwargs is not None:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, **attention_module_kwargs)
            else:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        else:
            self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.dynamic_weight_generator_q = DynamicWeightGenerator(d_model, d_model // 2, d_model)
        self.dynamic_weight_generator_k = DynamicWeightGenerator(d_model, d_model // 2, d_model)
        self.dynamic_weight_generator_v = DynamicWeightGenerator(d_model, d_model // 2, d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, input_gl=None, memory=None,
                isencoder=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        # 动态权重生成和应用
        dynamic_weights_q = self.dynamic_weight_generator_q(queries)
        dynamic_weights_k = self.dynamic_weight_generator_k(keys)
        dynamic_weights_v = self.dynamic_weight_generator_v(values)

        queries = queries * dynamic_weights_q
        keys = keys * dynamic_weights_k
        values = values * dynamic_weights_v

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights, input_gl=input_gl)
            out = queries + self.dropout(torch.relu(out))
        else:
            if isencoder:
                out = self.attention(queries, keys, values, attention_mask, attention_weights,
                                     input_gl=input_gl, memory=memory, isencoder=isencoder)
            else:
                out = self.attention(queries, keys, values, attention_mask, attention_weights,
                                     input_gl=None, memory=memory, isencoder=isencoder)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out