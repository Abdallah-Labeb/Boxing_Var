import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# الجزء الأول: Spatial Attention Module (تنفيذ يدوي لشرط رقم 6)
# الوظيفة: التركيز على "أماكن" معينة في الصورة (Where to look?)
# ---------------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # التأكد من حجم الكيرنل للحفاظ على أبعاد الصورة
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        # طبقة Convolution لدمج معلومات الماكس والمتوسط
        # In_channels = 2 (Max + Avg), Out_channels = 1 (Attention Map)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. حساب متوسط القنوات (Avg Pool Channel-wise)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 2. حساب أقصى قيمة للقنوات (Max Pool Channel-wise)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 3. دمجهم مع بعض (Concatenation)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # 4. استخراج خريطة الانتباه (Spatial Map)
        out = self.conv1(x_cat)
        
        # 5. تحويل القيم بين 0 و 1
        return self.sigmoid(out)

# ---------------------------------------------------------
# الجزء الثاني: الموديل الكامل (HitNet)
# Custom CNN + Attention
# ---------------------------------------------------------
class HitNet(nn.Module):
    def __init__(self):
        super(HitNet, self).__init__()
        
        # Block 1: استخراج الميزات الأولية
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Block 2: تعميق الشبكة
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3: استخراج ميزات أدق
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # إضافة الـ Spatial Attention (السحر كله هنا)
        self.attention = SpatialAttention(kernel_size=7)
        
        # Classifier (التصنيف النهائي)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # بيحول أي حجم لـ 1x1
        self.fc = nn.Linear(128, 2) # المخرج: قيمتين (Miss=0, Land=1)

    def forward(self, x):
        # مرحلة استخراج الميزات (Feature Extraction)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # تطبيق الانتباه (Attention Mechanism)
        # نضرب الميزات الأصلية في خريطة الانتباه
        # الأماكن المهمة (القفاز) هتزيد قيمتها، والخلفية هتقل
        att_map = self.attention(x)
        x = x * att_map
        
        # مرحلة التصنيف (Classification)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # تحويل المصفوفة لمتجه مسطح
        x = self.fc(x)
        
        return x
