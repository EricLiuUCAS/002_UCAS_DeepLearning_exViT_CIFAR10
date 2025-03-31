import torch
import torch.nn as nn

# Embedding层：将图像分成 patches，并利用卷积层映射到 embedding 维度
class EmbedLayer(nn.Module):
    def __init__(self, n_channels, embed_dim, image_size, patch_size, dropout=0.0):
        super(EmbedLayer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        )
        n_patches = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_patches, embed_dim), requires_grad=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = x + self.pos_embedding
        # 正确：在序列维度(即 dim=1)上拼接 cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + num_patches, embed_dim)
        x = self.dropout(x)
        return x


# 自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, n_attention_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_attention_heads = n_attention_heads
        self.head_embed_dim = embed_dim // n_attention_heads

        self.queries = nn.Linear(embed_dim, self.head_embed_dim * n_attention_heads)
        self.keys = nn.Linear(embed_dim, self.head_embed_dim * n_attention_heads)
        self.values = nn.Linear(embed_dim, self.head_embed_dim * n_attention_heads)
        self.out_projection = nn.Linear(self.head_embed_dim * n_attention_heads, embed_dim)

    def forward(self, x):
        b, s, e = x.shape
        # 分别计算 Q, K, V
        q = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim).transpose(1, 2)  # (b, head, s, dim)
        k = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim).transpose(1, 2)
        v = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim).transpose(1, 2)
        # 计算注意力得分
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_embed_dim ** 0.5)
        attn = torch.softmax(attn_scores, dim=-1)
        x = torch.matmul(attn, v)
        # 重新组合
        x = x.transpose(1, 2).reshape(b, s, e)
        x = self.out_projection(x)
        return x

# Transformer Encoder 层
class Encoder(nn.Module):
    def __init__(self, embed_dim, n_attention_heads, forward_mul, dropout=0):
        super(Encoder, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, n_attention_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim * forward_mul)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * forward_mul, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.attention(self.norm1(x)))
        x = x + self.dropout2(self.fc2(self.activation(self.fc1(self.norm2(x)))))
        return x


# 分类器：使用 cls token 进行分类
class Classifier(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        # 取 cls token 对应的 embedding
        x = x[:, 0, :]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# Vision Transformer（自实现 Encoder）
class VisionTransformer(nn.Module):
    def __init__(self, n_channels, embed_dim, n_layers, n_attention_heads,
                 forward_mul, image_size, patch_size, n_classes, dropout=0):
        super(VisionTransformer, self).__init__()
        self.embedding = EmbedLayer(n_channels, embed_dim, image_size, patch_size, dropout=dropout)
        self.encoder = nn.ModuleList([
            Encoder(embed_dim, n_attention_heads, forward_mul, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = Classifier(embed_dim, n_classes)
        self.apply(vit_init_weights)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.encoder:
            x = layer(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x

# Vision Transformer 使用 torch 内置 TransformerEncoder
class VisionTransformer_pytorch(nn.Module):
    def __init__(self, n_channels, embed_dim, n_layers, n_attention_heads,
                 forward_mul, image_size, patch_size, n_classes, dropout=0.1):
        super(VisionTransformer_pytorch, self).__init__()
        self.embedding = EmbedLayer(n_channels, embed_dim, image_size, patch_size, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_attention_heads,
            dim_feedforward=forward_mul * embed_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers, norm=nn.LayerNorm(embed_dim))
        self.classifier = Classifier(embed_dim, n_classes)
        self.apply(vit_init_weights)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.classifier(x)
        return x

# 初始化函数
def vit_init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, mean=0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, EmbedLayer):
        nn.init.trunc_normal_(m.cls_token, mean=0, std=0.02)
        nn.init.trunc_normal_(m.pos_embedding, mean=0, std=0.02)
