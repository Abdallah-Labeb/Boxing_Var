import torch
import torch.nn as nn
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import shutil
from PIL import Image

# -A. ActionNet Components
class CustomCNN(nn.Module):
    def __init__(self, out_dim=512):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, out_dim)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x).flatten(1) 
        return self.fc(x)

class HandCodedSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(HandCodedSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [self.queries(queries), self.keys(keys)])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, self.values(values)]).reshape(N, query_len, self.heads * self.head_dim)
        return self.fc_out(out)

class ActionNet(nn.Module):
    def __init__(self, num_classes=3, latent_dim=512, num_heads=8):
        super(ActionNet, self).__init__()
        self.backbone = CustomCNN(out_dim=latent_dim)
        self.attention = HandCodedSelfAttention(embed_size=latent_dim, heads=num_heads)
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, num_classes))
    
    def forward(self, x):
        batch_size, C, frames, H, W = x.shape
        c_in = x.reshape(batch_size * frames, C, H, W)
        features = self.backbone(c_in).reshape(batch_size, frames, -1)
        attn_out = self.attention(features, features, features, mask=None)
        transformer_out = self.transformer_encoder(attn_out + features)
        return self.classifier(torch.mean(transformer_out, dim=1))

# --- B. HitNet Components (مع التصحيحات الهيكلية) ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # تصحيح: استخدام Linear بدلاً من Conv2d ليتطابق مع الأوزان في الملف
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_in = self.avg_pool(x).view(b, c)
        max_in = self.max_pool(x).view(b, c)
        
        avg_out = self.fc(avg_in).view(b, c, 1, 1)
        max_out = self.fc(max_in).view(b, c, 1, 1)
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(planes)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        out = self.channel_att(x) * x
        out = self.spatial_att(out) * out
        return out

class HitNet(nn.Module):
    def __init__(self):
        super(HitNet, self).__init__()
        # تصحيح: إضافة Dropout لتطابق ترتيب الطبقات في ملف الأوزان
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout(0.5), nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.5), nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout(0.5), nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout(0.5), nn.MaxPool2d(2)
        )
        self.attention = CBAM(256)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.attention(x)
        x = torch.mean(x, dim=[2, 3]) # GAP
        return self.classifier(x)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_CLASSES = ['Jab', 'Cross', 'Hook']
HIT_CLASSES = ['Miss', 'Land']

print(f"⏳ Loading Models on {DEVICE}...")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACTION_MODEL_PATH = os.path.join(SCRIPT_DIR, "ActionNet_Custom_Final.pth")
HIT_MODEL_PATH = os.path.join(SCRIPT_DIR, "HitNet_Final.pth")

# Load ActionNet
action_model = ActionNet(num_classes=3).to(DEVICE)
if os.path.exists(ACTION_MODEL_PATH):
    try:
        action_model.load_state_dict(torch.load(ACTION_MODEL_PATH, map_location=DEVICE))
        action_model.eval()
        print("✅ ActionNet Loaded.")
    except Exception as e:
        print(f"❌ Error Loading ActionNet: {e}")
else:
    print(f"❌ Error: ActionNet_Custom_Final.pth not found at {ACTION_MODEL_PATH}")

# Load HitNet
hit_model = HitNet().to(DEVICE)
if os.path.exists(HIT_MODEL_PATH):
    try:
        # strict=False أحياناً يفيد لتجاهل مفاتيح غير مهمة، لكن سنتركه True للتأكد
        hit_model.load_state_dict(torch.load(HIT_MODEL_PATH, map_location=DEVICE))
        hit_model.eval()
        print("✅ HitNet Loaded.")
    except Exception as e:
        print(f"❌ Error Loading HitNet: {e}")
        print("💡 Hint: If mismatch persists, try setting strict=False in load_state_dict.")
else:
    print(f"❌ Error: HitNet_Final.pth not found at {HIT_MODEL_PATH}")


# --- Transforms ---
# هام: يجب استخدام نفس Normalization المستخدمة في التدريب
# ImageNet normalization (الأكثر شيوعاً)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Minimum confidence threshold - لو أقل من هذا يعتبر "Unknown"
MIN_CONFIDENCE_THRESHOLD = 0.4


def process_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    original_frames = [] 
    
    # قراءة الفريمات
    while True:
        ret, frame = cap.read()
        if not ret: break
        original_frames.append(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(frame_rgb))
        # نأخذ أول 60 فريم فقط لعدم ملء الذاكرة
        if len(frames) >= 60: 
            break
            
    cap.release()

    if len(frames) == 0: return None, None, None

    # تحسين الـ Padding: لو الفيديو أقصر من 16، نكرره (Loop) بدلاً من تكرار آخر صورة فقط
    final_frames = []
    target_len = 16
    
    if len(frames) >= target_len:
        # لو الفيديو طويل، نأخذ عينات متساوية (Uniform Sampling)
        indices = np.linspace(0, len(frames) - 1, target_len).astype(int)
        final_frames = [frames[i] for i in indices]
    else:
        # لو الفيديو قصير، نكرره
        final_frames = frames.copy()
        while len(final_frames) < target_len:
            # نأخذ من البداية (Loop)
            needed = target_len - len(final_frames)
            final_frames.extend(frames[:needed])
    
    # تحضير Action Tensor
    video_tensor = torch.stack(final_frames).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)
    
    # تحضير Hit Tensors - نأخذ عدة فريمات بدلاً من واحد فقط
    # نأخذ 5 فريمات موزعة على الفيديو (25%, 40%, 50%, 60%, 75%)
    hit_indices = [int(len(original_frames) * p) for p in [0.25, 0.4, 0.5, 0.6, 0.75]]
    hit_indices = [min(i, len(original_frames) - 1) for i in hit_indices]  # تجنب تجاوز الحدود
    
    hit_tensors = []
    for idx in hit_indices:
        frame = original_frames[idx]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hit_tensors.append(transform(frame_rgb))
    
    hit_batch = torch.stack(hit_tensors).to(DEVICE)  # Shape: (5, 3, 224, 224)
    
    return video_tensor, hit_batch, len(original_frames)


@app.get("/")
def home():
    return {"message": "Boxing AI API is Running! 🥊"}

@app.get("/debug")
def debug_info():
    """Endpoint للتشخيص - يعرض معلومات عن النماذج"""
    return {
        "device": str(DEVICE),
        "action_model_loaded": action_model is not None,
        "hit_model_loaded": hit_model is not None,
        "action_classes": ACTION_CLASSES,
        "hit_classes": HIT_CLASSES,
        "min_confidence_threshold": MIN_CONFIDENCE_THRESHOLD
    }

@app.post("/predict")
async def predict_punch(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        video_tensor, hit_batch, total_frames = process_video_frames(temp_filename)
        
        if video_tensor is None:
            return {"error": "Could not process video. Maybe file is corrupted?"}
        
        print(f"📊 Video Info: {total_frames} frames processed")
        
        with torch.no_grad():
            # === Action Prediction ===
            action_out = action_model(video_tensor)
            action_probs = torch.softmax(action_out, dim=1)[0]
            action_idx = torch.argmax(action_probs).item()
            action_conf = action_probs[action_idx].item()
            
            # Debug: طباعة كل الاحتمالات
            print(f"🥊 Action Probs: Jab={action_probs[0]:.3f}, Cross={action_probs[1]:.3f}, Hook={action_probs[2]:.3f}")
            
            # التحقق من الثقة
            if action_conf < MIN_CONFIDENCE_THRESHOLD:
                action_name = "Unknown"
                print(f"⚠️ Low confidence for action: {action_conf:.3f}")
            else:
                action_name = ACTION_CLASSES[action_idx]
            
            # === Hit Prediction (Multi-frame voting) ===
            # نمرر كل الفريمات ونأخذ المتوسط
            hit_out = hit_model(hit_batch)  # Shape: (5, 2)
            hit_probs_all = torch.softmax(hit_out, dim=1)  # Shape: (5, 2)
            
            # طريقة التصويت: متوسط الاحتمالات من كل الفريمات
            hit_probs = torch.mean(hit_probs_all, dim=0)  # Shape: (2,)
            hit_idx = torch.argmax(hit_probs).item()
            hit_conf = hit_probs[hit_idx].item()
            
            # Debug: طباعة تفاصيل Hit
            print(f"🎯 Hit Probs (averaged): Miss={hit_probs[0]:.3f}, Land={hit_probs[1]:.3f}")
            print(f"🎯 Individual frame predictions: {hit_probs_all.cpu().numpy()}")
            
            # التحقق من الثقة
            if hit_conf < MIN_CONFIDENCE_THRESHOLD:
                hit_name = "Uncertain"
                print(f"⚠️ Low confidence for hit: {hit_conf:.3f}")
            else:
                hit_name = HIT_CLASSES[hit_idx]
            
        return {
            "punch_type": action_name,
            "punch_confidence": f"{action_conf*100:.1f}%",
            "hit_result": hit_name,
            "hit_confidence": f"{hit_conf*100:.1f}%",
            "message": f"Detected a {action_name} that was a {hit_name}!",
            "debug": {
                "action_probs": [f"{p:.3f}" for p in action_probs.cpu().tolist()],
                "hit_probs": [f"{p:.3f}" for p in hit_probs.cpu().tolist()],
                "total_frames": total_frames
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc() # طباعة الخطأ في الكونسول للمطور
        return {"error": str(e)}

    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)