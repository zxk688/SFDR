import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from model.transformer.attention import MultiHeadAttention
from model.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
from model.containers import Module, ModuleList


class MeshedDecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)


    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):  #
        self_att = self.self_att(input, input, input, mask_self_att, input_gl=None,
                                 isencoder=True)
        self_att = self_att * mask_pad

        enc_att1 = self.enc_att(self_att, enc_output[:, 0], enc_output[:, 0], mask_enc_att) * mask_pad
        enc_att2 = self.enc_att(self_att, enc_output[:, 1], enc_output[:, 1], mask_enc_att) * mask_pad
        enc_att3 = self.enc_att(self_att, enc_output[:, 2], enc_output[:, 2], mask_enc_att) * mask_pad


        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([self_att, enc_att1], -1)))
        alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([self_att, enc_att2], -1)))
        alpha3 = torch.sigmoid(self.fc_alpha3(torch.cat([self_att, enc_att3], -1)))

        enc_att = (enc_att1 * alpha1 + enc_att2 * alpha2 + enc_att3 * alpha3) / np.sqrt(3)

        enc_att = enc_att * mask_pad

        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff


class MeshedDecoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoder, self).__init__()
        self.d_model = d_model  # 512
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)  # 283,512
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len+1, d_model, 0),
                                                    freeze=True)  # 128,512
        self.layers = ModuleList(
            [MeshedDecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder):
        # input (b_s, seq_len) 10 30     #   10 30; 10 3 50 512; 10 1 1 50

        b_s, seq_len = input.shape[:2]  # 10 15
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)  10 30 1  ; 10 15 1
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)  # 30 30; 15 15
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(
            0)  # (1, 1, seq_len, seq_len) 1 1 30 30; 1 1 15 15
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(
            1).byte()  # 10 1 30 30; 10 1 15 15
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len) 10 1 30 30; 10 1 15 15

        if self._is_stateful:  # false
            device = self.running_mask_self_attention.device
            mask_self_attention = torch.tensor(mask_self_attention, dtype=torch.uint8).to(device)
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention],
                                                             -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len) 10 30; 10 15
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)  # 10 30; 10 15

        if self._is_stateful:  # false type(seq)
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)  # 10 30 512; 10 15 512     50,1,512

        for i, l in enumerate(self.layers):
            out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)  # 转到MeshedDecoderLayer
            # 10 30 512; 10 3 50 512; 10 30 1; 10 1 30 30 ; 10 1 1 50     ##

        out = self.fc(out)  # 10, 15, 10199     # 50,1,283
        return F.log_softmax(out, dim=-1)  # torch.Size([10, 27, 14130])
