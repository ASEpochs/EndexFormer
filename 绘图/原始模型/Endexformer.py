import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.Endexformer_EncDec import FlattenHead,EncoderLayer,Encoder,EnEmbedding
import numpy as np




class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        # 保证MS时n_vars为1，其他情况下为configs.enc_in
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in

        # Embedding Layers
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder Head
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout)

    def _normalize(self, x_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            return x_enc, means, stdev
        return x_enc, None, None

    def _denormalize(self, dec_out, means, stdev):
        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def _get_embeddings(self, x_enc, x_mark_enc):
        # 修改embedding部分，根据features决定是否对输入进行变换
        if self.features == 'MS':
            en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))  # 只使用最后一列
        else:
            en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))  # 对于M特征，使用整个输入
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)
        return en_embed, ex_embed, n_vars

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc, means, stdev = self._normalize(x_enc)

        en_embed, ex_embed, n_vars = self._get_embeddings(x_enc, x_mark_enc)
        
        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]).permute(0, 1, 3, 2)

        dec_out = self.head(enc_out).permute(0, 2, 1)
        dec_out = self._denormalize(dec_out, means, stdev)

        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc, means, stdev = self._normalize(x_enc)

        en_embed, ex_embed, n_vars = self._get_embeddings(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]).permute(0, 1, 3, 2)

        dec_out = self.head(enc_out).permute(0, 2, 1)
        dec_out = self._denormalize(dec_out, means, stdev)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            if self.features == 'M':
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # 返回预测结果的最后 pred_len 长度部分
        return None
