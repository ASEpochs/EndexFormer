import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import  PositionalEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


# class EnEmbedding(nn.Module):
#     def __init__(self, n_vars, d_model, patch_len, dropout):
#         super(EnEmbedding, self).__init__()
#         # Patching
#         self.patch_len = patch_len

#         self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
#         self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
#         self.position_embedding = PositionalEmbedding(d_model)

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         # do patching
#         n_vars = x.shape[1]
#         glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

#         x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
#         x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
#         # Input encoding
#         x = self.value_embedding(x) + self.position_embedding(x)
#         x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
#         x = torch.cat([x, glb], dim=2)
#         x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
#         return self.dropout(x), n_vars


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat(x.shape[0], 1, 1, 1)

        # Unfold the input tensor and reshape it
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])

        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)

        # Reshape for the next operation
        x = x.reshape(-1, n_vars, x.shape[-2], x.shape[-1])

        # Concatenate global token and reshape again
        x = torch.cat([x, glb], dim=2)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])

        return self.dropout(x), n_vars







class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


# class EncoderLayer(nn.Module):
#     def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
#                  dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.self_attention = self_attention
#         self.cross_attention = cross_attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
#         B, L, D = cross.shape
#         x = x + self.dropout(self.self_attention(
#             x, x, x,
#             attn_mask=x_mask,
#             tau=tau, delta=None
#         )[0])
#         x = self.norm1(x)

#         x_glb_ori = x[:, -1, :].unsqueeze(1)
#         x_glb = torch.reshape(x_glb_ori, (B, -1, D))
#         x_glb_attn = self.dropout(self.cross_attention(
#             x_glb, cross, cross,
#             attn_mask=cross_mask,
#             tau=tau, delta=delta
#         )[0])
#         x_glb_attn = torch.reshape(x_glb_attn,
#                                    (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
#         x_glb = x_glb_ori + x_glb_attn
#         x_glb = self.norm2(x_glb)

#         y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))

#         return self.norm3(x + y)


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention

        # 使用 nn.Conv1d 时，in_channels 和 out_channels 直接传递变量
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)

        # 将 LayerNorm 和 Dropout 直接初始化为类属性
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 使用字典的方式选择激活函数，更易于扩展
        activation_funcs = {"relu": F.relu, "gelu": F.gelu}
        self.activation = activation_funcs.get(activation, F.relu)  # 默认使用 ReLU

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        # 获取 cross 的形状信息
        B, L, D = cross.shape

        # 1. 自注意力分支：计算自注意力并进行残差连接
        self_attn_out = self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        x = x + self.dropout(self_attn_out)
        x = self.norm1(x)

        # 2. 全局 token 分支：
        # 从 x 中提取最后一个 token 作为全局 token（形状为 [B, 1, D]）
        global_token = x[:, -1:, :]

        # 保持形状不变，此处使用 reshape 重构形状（效果等同于 global_token）
        global_token_query = torch.reshape(global_token, (B, -1, D))

        # 利用全局 token 作为 query，对 cross 执行交叉注意力计算
        cross_attn = self.cross_attention(
            global_token_query, cross, cross,
            attn_mask=cross_mask, tau=tau, delta=delta
        )[0]
        cross_attn = self.dropout(cross_attn)

        # 将交叉注意力结果重新调整形状为 [B, 1, D]
        cross_attn = torch.reshape(
            cross_attn, (cross_attn.shape[0] * cross_attn.shape[1], cross_attn.shape[2])
        ).unsqueeze(1)

        # 将原始全局 token 与交叉注意力结果相加，并进行归一化
        global_token_updated = self.norm2(global_token + cross_attn)

        # 3. 拼接：用更新后的全局 token 替换 x 中的最后一个 token
        x_combined = torch.cat([x[:, :-1, :], global_token_updated], dim=1)

        # 4. 前馈卷积层
        # 注意：conv1/conv2 接受的维度需要是 (B, D, L)，因此需要先进行转置
        ff = self.conv1(x_combined.transpose(-1, 1))
        ff = self.activation(ff)
        ff = self.dropout(ff)
        ff = self.conv2(ff).transpose(-1, 1)
        ff = self.dropout(ff)

        # 5. 残差连接并最终归一化
        output = self.norm3(x_combined + ff)
        return output
