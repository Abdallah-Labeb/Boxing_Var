from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import os
import shutil
import io
import math
from PIL import Image


# 1. تعريف معمارية الموديلات


# --- A. ActionTransformer Components (for best_model.pth) ---

class AttentionModule(nn.Module):
    def __init__(self, embed_size, heads):
        super(AttentionModule, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        return self.fc_out(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x): 
        return x + self.pe[:, :x.size(1)]


class ActionTransformer(nn.Module):
    def __init__(self, num_classes=4):
        super(ActionTransformer, self).__init__()
    
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.feature_dim = 256
        self.pos_encoder = PositionalEncoding(self.feature_dim)
       
        self.custom_attention = AttentionModule(embed_size=256, heads=4)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.cnn(x).flatten(1).view(b, t, -1)
        x = self.pos_encoder(x)
        
        # Custom attention
        attention_out = self.custom_attention(values=x, keys=x, query=x)
        x = x + attention_out
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        x = x.mean(dim=1)            
        return self.fc(x)

# --- B. HitNet Components (CBAM Attention) ---

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        
        return x * attention


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class HitNet(nn.Module):
    def __init__(self, dropout_rate=0.5, attention_type='cbam'):
        super(HitNet, self).__init__()
        
        self.attention_type = attention_type
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )
        
        if attention_type == 'cbam':
            self.attention = CBAM(channels=256, reduction=16)
        elif attention_type == 'channel':
            self.attention = ChannelAttention(channels=256, reduction=16)
        elif attention_type == 'spatial':
            self.attention = SpatialAttention(kernel_size=7)
        else:
            self.attention = None
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        if self.attention is not None:
            x = self.attention(x)
        x = self.global_pool(x).flatten(1)
        out = self.classifier(x)
        return out

# ==========================================
# 2. إعداد الـ API وتحميل الموديلات
# ==========================================
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Classes for ActionTransformer (best_model.pth)
ACTION_CLASSES = ['Head_Left', 'Head_Right', 'Body_Left', 'Body_Right']
HIT_CLASSES = ['Miss', 'Land']

# Get the directory where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"⏳ Loading Models on {DEVICE}...")

# Load ActionTransformer (best_model.pth)
action_model = ActionTransformer(num_classes=4).to(DEVICE)
action_model_path = os.path.join(BASE_DIR, "best_model.pth")
if os.path.exists(action_model_path):
    try:
        checkpoint = torch.load(action_model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            action_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            action_model.load_state_dict(checkpoint)
        action_model.eval()
        print("✅ ActionTransformer Loaded.")
    except Exception as e:
        print(f"❌ Error loading ActionTransformer: {e}")
else:
    print(f"❌ Error: best_model.pth not found at {action_model_path}")

# Load HitNet
hit_model = HitNet(dropout_rate=0.5, attention_type='cbam').to(DEVICE)
hit_model_path = os.path.join(BASE_DIR, "HitNet_Final.pth")
if os.path.exists(hit_model_path):
    # Try loading as state_dict first, if fails try as checkpoint
    try:
        checkpoint = torch.load(hit_model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            hit_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            hit_model.load_state_dict(checkpoint)
        hit_model.eval()
        print("✅ HitNet Loaded.")
    except Exception as e:
        print(f"❌ Error loading HitNet: {e}")
else:
    print(f"❌ Error: HitNet_Final.pth not found at {hit_model_path}")


# Transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. دالة معالجة الفيديو واستخراج النتائج
# ==========================================
def process_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    original_frames = [] 
    while len(frames) < 16:
        ret, frame = cap.read()
        if not ret: break
        original_frames.append(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(frame_rgb))
    cap.release()

    if len(frames) == 0: return None, None

    # Padding
    while len(frames) < 16:
        frames.append(frames[-1])
        original_frames.append(original_frames[-1])
    
    # Prepare Action Tensor (C, T, H, W)
    video_tensor = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)
    
    # Prepare Hit Image (Take middle frame)
    mid_idx = len(original_frames) // 2
    mid_frame = original_frames[mid_idx]
    mid_frame_rgb = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2RGB)
    hit_tensor = transform(mid_frame_rgb).unsqueeze(0).to(DEVICE)
    
    return video_tensor, hit_tensor

# ==========================================
# 4. الـ Endpoints
# ==========================================

@app.get("/")
def home():
    return {"message": "Boxing AI API is Running! 🥊"}

@app.post("/predict")
async def predict_punch(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        video_tensor, hit_tensor = process_video_frames(temp_filename)
        
        if video_tensor is None:
            return {"error": "Could not process video"}
        
        with torch.no_grad():
            # ActionNet Inference
            action_out = action_model(video_tensor)
            action_idx = torch.argmax(action_out, dim=1).item()
            action_name = ACTION_CLASSES[action_idx]
            action_conf = torch.softmax(action_out, dim=1)[0][action_idx].item()
            
            # HitNet Inference
            hit_out = hit_model(hit_tensor)
            hit_idx = torch.argmax(hit_out, dim=1).item()
            hit_name = HIT_CLASSES[hit_idx]
            hit_conf = torch.softmax(hit_out, dim=1)[0][hit_idx].item()
            
        return {
            "punch_type": action_name,
            "punch_confidence": f"{action_conf*100:.1f}%",
            "hit_result": hit_name,
            "hit_confidence": f"{hit_conf*100:.1f}%",
            "message": f"Detected a {action_name} that was a {hit_name}!"
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)