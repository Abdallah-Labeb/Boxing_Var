# 🥊 BoxingAI - Real-Time Boxing Punch Analyzer

<div align="center">

![BoxingAI Banner](https://img.shields.io/badge/BoxingAI-v2.1-00f3ff?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)

**An AI-powered system for analyzing boxing footage using deep learning**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-reference) • [Models](#-model-architecture)

</div>

---

## 📖 Overview

BoxingAI is a cutting-edge computer vision system that analyzes boxing videos to:
- **Detect punch types** (Jab, Cross, Hook)
- **Determine if punches landed** (Land/Miss)

The system uses two custom neural networks:
- **ActionNet**: A CNN + Transformer hybrid for temporal action recognition
- **HitNet**: A CNN with CBAM attention for impact detection

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Punch Classification** | Identifies Jab, Cross, and Hook punches |
| 💥 **Impact Detection** | Determines if a punch landed or missed |
| 🎬 **Video Analysis** | Processes video clips frame-by-frame |
| ⚡ **Real-time API** | FastAPI backend for instant predictions |
| 🖥️ **Modern UI** | Futuristic cyberpunk-style interface |
| 🔧 **GPU Support** | CUDA acceleration for faster inference |

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA (optional, for GPU acceleration)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/BoxingAI.git
cd BoxingAI
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install fastapi uvicorn opencv-python pillow numpy python-multipart
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### Step 4: Download Model Weights

Place the following model files in `backend/models/`:
- `ActionNet_Custom_Final.pth` - Punch type classifier
- `HitNet_Final.pth` - Impact detector

---

## 🚀 Usage

### Starting the Backend Server

```bash
cd backend/models
python main.py
```

The API server will start at `http://localhost:8000`

### Opening the Frontend

Simply open `index.html` in your browser, or use a local server:

```bash
# Using Python
python -m http.server 3000

# Then open http://localhost:3000
```

### Using the Interface

1. Click **"LOAD FOOTAGE"** to select a boxing video
2. Click **"INITIATE ANALYSIS"** to process the video
3. View results in the **DIAGNOSTICS** panel

---

## 📡 API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

#### `GET /`
Health check endpoint

**Response:**
```json
{
    "message": "Boxing AI API is Running! 🥊"
}
```

#### `GET /debug`
Get model and system information

**Response:**
```json
{
    "device": "cuda",
    "action_model_loaded": true,
    "hit_model_loaded": true,
    "action_classes": ["Jab", "Cross", "Hook"],
    "hit_classes": ["Miss", "Land"],
    "min_confidence_threshold": 0.4
}
```

#### `POST /predict`
Analyze a boxing video

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` - Video file (mp4, avi, mov)

**Response:**
```json
{
    "punch_type": "Hook",
    "punch_confidence": "95.2%",
    "hit_result": "Land",
    "hit_confidence": "87.3%",
    "message": "Detected a Hook that was a Land!",
    "debug": {
        "action_probs": ["0.020", "0.028", "0.952"],
        "hit_probs": ["0.127", "0.873"],
        "total_frames": 45
    }
}
```

---

## 🧠 Model Architecture

### ActionNet (Punch Type Classification)

```
┌─────────────────────────────────────────────────────────┐
│                      Input Video                         │
│                   (B, 3, 16, 224, 224)                   │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    CustomCNN Backbone                    │
│   Conv2d → BN → ReLU → MaxPool (×4 blocks)              │
│   Output: 512-dim feature per frame                      │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Hand-Coded Self-Attention                   │
│   Multi-head attention (8 heads) on temporal sequence   │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Transformer Encoder (2 layers)              │
│   Captures long-range temporal dependencies             │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  Classifier Head                         │
│   Dropout → Linear(512,256) → ReLU → Linear(256,3)     │
│   Output: [Jab, Cross, Hook] probabilities              │
└─────────────────────────────────────────────────────────┘
```

### HitNet (Impact Detection)

```
┌─────────────────────────────────────────────────────────┐
│                    Input Frames                          │
│                   (B, 3, 224, 224)                       │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   CNN Backbone                           │
│   Conv2d → BN → ReLU → Dropout → MaxPool (×4 blocks)   │
│   Output: (B, 256, 14, 14)                              │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              CBAM Attention Module                       │
│   ┌─────────────────────────────────────┐               │
│   │    Channel Attention (Squeeze)      │               │
│   │    Avg Pool + Max Pool → MLP        │               │
│   └─────────────────┬───────────────────┘               │
│                     │                                    │
│   ┌─────────────────▼───────────────────┐               │
│   │    Spatial Attention                │               │
│   │    Avg + Max → Conv2d(7×7)          │               │
│   └─────────────────────────────────────┘               │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Global Average Pooling                      │
│   (B, 256, 14, 14) → (B, 256)                           │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  Classifier Head                         │
│   Dropout → Linear(256,128) → ReLU → Dropout            │
│   → Linear(128,64) → ReLU → Linear(64,2)               │
│   Output: [Miss, Land] probabilities                    │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
GUI_CNN/
├── 📄 index.html              # Frontend UI
├── 📄 README.md               # This file
├── 📄 requirements.txt        # Python dependencies
│
└── 📁 backend/
    └── 📁 models/
        ├── 📄 main.py                    # FastAPI server & model definitions
        ├── 📦 ActionNet_Custom_Final.pth # Trained ActionNet weights
        ├── 📦 HitNet_Final.pth           # Trained HitNet weights
        └── 📁 uploads/                   # Temporary upload folder
```

---

## ⚙️ Configuration

### Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MIN_CONFIDENCE_THRESHOLD` | 0.4 | Minimum confidence for valid prediction |
| `target_len` | 16 | Number of frames for ActionNet |
| `image_size` | 224×224 | Input image dimensions |

### Normalization (ImageNet)

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Model not loading | Check if `.pth` files exist in `backend/models/` |
| CUDA out of memory | Reduce video length or use CPU |
| Same prediction always | Check model weights & normalization |
| CORS error | Ensure backend is running on port 8000 |

### Debug Mode

Access `http://localhost:8000/debug` to check:
- Device (CPU/CUDA)
- Model loading status
- Configuration values

---

## 📊 Performance

| Metric | ActionNet | HitNet |
|--------|-----------|--------|
| Input Size | 16×224×224 | 224×224 |
| Parameters | ~5M | ~2M |
| Inference Time (GPU) | ~50ms | ~10ms |
| Inference Time (CPU) | ~200ms | ~50ms |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- FastAPI for the blazing-fast API framework
- CBAM paper authors for the attention mechanism

---

<div align="center">

**Made with ❤️ and 🥊**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/BoxingAI?style=social)](https://github.com/yourusername/BoxingAI)

</div>
