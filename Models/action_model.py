import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------------------------------------------
# الجزء الأول: تنفيذ الـ Attention يدويًا (شرط رقم 6)
# ---------------------------------------------------------
class ManualSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(ManualSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # تعريف المصفوفات الخطية للـ Query, Key, Value
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # عدد الأمثلة في الباتش
        N = query.shape[0]
        # أطوال السيكوينس (عدد الفريمات)
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # تقسيم الإدخال لعدة Heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # 1. معادلة الـ Attention: (Q * K^T)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # 2. القسمة على جذر البعد (Scaling)
        # Scale Dot-Product Attention
        energy = energy / (self.embed_size ** (1 / 2))

        # 3. تطبيق الـ Softmax للحصول على الاحتمالات
        attention = torch.softmax(energy, dim=3)

        # 4. الضرب في الـ Values
        out = torch.einsum("nhqk,nvhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

# ---------------------------------------------------------
# الجزء الثاني: بناء Transformer Block
# ---------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = ManualSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # Skip Connection 1 + Attention
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))

        # Skip Connection 2 + Feed Forward
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# ---------------------------------------------------------
# الجزء الثالث: الموديل النهائي (CNN + Transformer)
# ---------------------------------------------------------
class ActionTransformer(nn.Module):
    def __init__(
        self,
        num_classes=3,   # عدد الحركات (Jab, Cross, Hook)
        num_frames=16,   # طول الفيديو
        embed_size=256,  # حجم الفيكتور
        num_layers=2,    # عدد طبقات الترانزفورمر
        heads=4,
        forward_expansion=4,
        dropout=0.1
    ):
        super(ActionTransformer, self).__init__()

        # 1. CNN Feature Extractor (Backbone)
        # بدلاً من استخدام ResNet جاهز، سنبني واحد بسيط يدويًا
        # ليحول الصورة (3, 224, 224) إلى فيكتور مميز
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # -> 112x112
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 56x56

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> 7x7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # -> 1x1 (Feature Vector)
        )

        # طبقة لضبط حجم الفيكتور ليدخل للترانزفورمر
        self.feature_proj = nn.Linear(64, embed_size)

        # Positional Encoding (عشان يعرف ترتيب الفريمات)
        self.position_embedding = nn.Embedding(num_frames, embed_size)

        # طبقات الترانزفورمر
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        # الطبقة الأخيرة للتصنيف
        self.fc_out = nn.Linear(embed_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Channels, Frames, Height, Width)
        # مثال: (2, 3, 16, 224, 224)

        b, c, t, h, w = x.shape

        # دمج الباتش مع الفريمات عشان ندخلهم الـ CNN كلهم مرة واحدة
        # (Batch * Frames, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        # استخراج الميزات من كل فريم باستخدام الـ CNN
        features = self.cnn_backbone(x) # -> (B*T, 64, 1, 1)
        features = features.view(b, t, -1) # -> (B, T, 64)

        # تكبير الحجم لـ Embedding Size
        embeddings = self.feature_proj(features) # -> (B, T, 256)

        # إضافة معلومة الموقع (Positional Encoding)
        positions = torch.arange(0, t).expand(b, t).to(x.device)
        embeddings = embeddings + self.position_embedding(positions)

        out = self.dropout(embeddings)

        # المرور على طبقات الترانزفورمر
        for layer in self.layers:
            out = layer(out, out, out, mask=None)

        # التصنيف بناءً على متوسط الفريمات
        out = out.mean(dim=1)
        out = self.fc_out(out)

        return out
