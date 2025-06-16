
import math
import torch
from torch import nn
import torch.nn.functional as F
import copy

class DynamicWeightGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DynamicWeightGenerator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, features):
        x = self.relu(self.fc1(features))
        weights = torch.sigmoid(self.fc2(x))  # 生成的动态权重
        return weights



class GATopt(object):
    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = 8
        self.hidden_dropout_prob = 0.2
        self.attention_probs_dropout_prob = 0.2



class GAT(nn.Module):
    def __init__(self, config_gat):
        super(GAT, self).__init__()
        layer = GATLayer(config_gat)
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(config_gat.num_layers)])

    def forward(self, input_graph, dynamic_weights=None):
        hidden_states = input_graph
        for i, layer_module in enumerate(self.encoder):
            if dynamic_weights is not None:
                hidden_states = layer_module(hidden_states, dynamic_weights[:, i])
            else:
                hidden_states = layer_module(hidden_states)
        return hidden_states



class GAT_MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(GAT_MultiHeadAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_graph, dynamic_weights=None):
        nodes_q = self.query(input_graph)
        nodes_k = self.key(input_graph)
        nodes_v = self.value(input_graph)

        nodes_q_t = self.transpose_for_scores(nodes_q)
        nodes_k_t = self.transpose_for_scores(nodes_k)
        nodes_v_t = self.transpose_for_scores(nodes_v)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(nodes_q_t, nodes_k_t.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 应用邻接矩阵来限制注意力得分
        # attention_scores = attention_scores * adjacency_matrix.unsqueeze(1).unsqueeze(2)



        if dynamic_weights is not None:
            attention_scores = attention_scores * dynamic_weights.unsqueeze(1).unsqueeze(2)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        nodes_new = torch.matmul(attention_probs, nodes_v_t)
        nodes_new = nodes_new.permute(0, 2, 1, 3).contiguous()
        new_nodes_shape = nodes_new.size()[:-2] + (self.all_head_size,)
        nodes_new = nodes_new.view(*new_nodes_shape)

        return nodes_new


class GATLayer(nn.Module):
    def __init__(self, config):
        super(GATLayer, self).__init__()
        self.mha = GAT_MultiHeadAttention(config)

        self.fc_in = nn.Linear(config.hidden_size, config.hidden_size)
        self.bn_in = nn.BatchNorm1d(config.hidden_size)
        self.dropout_in = nn.Dropout(config.hidden_dropout_prob)

        self.fc_out = nn.Linear(config.hidden_size, config.hidden_size)
        self.bn_out = nn.BatchNorm1d(config.hidden_size)
        self.dropout_out = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_graph, dynamic_weights=None):
        attention_output = self.mha(input_graph, dynamic_weights)  # Multi-Head Attention

        # 自动构建场景图: 使用阈值来选择保留的边
        with torch.no_grad():
            attention_scores = torch.matmul(input_graph, input_graph.transpose(-1, -2))
            max_score, _ = attention_scores.max(dim=-1, keepdim=True)
            min_score, _ = attention_scores.min(dim=-1, keepdim=True)
            threshold = min_score + 0.5 * (max_score - min_score)  # 选择中间值
            adjacency_matrix = (attention_scores > threshold).float()  # 阈值筛选，得到邻接矩阵

        # 如果有动态权重，应用它们
        if dynamic_weights is not None:
            attention_output = attention_output * dynamic_weights.unsqueeze(-1)

        # 使用邻接矩阵进行特征强化
        attention_output = torch.matmul(adjacency_matrix, attention_output)

        attention_output = self.fc_in(attention_output)
        attention_output = self.dropout_in(attention_output)
        attention_output = self.bn_in((attention_output + input_graph).permute(0, 2, 1)).permute(0, 2, 1)

        return attention_output
