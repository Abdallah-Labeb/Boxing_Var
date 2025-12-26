import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # شريط تحميل عشان نشوف التقدم
import os

# كلاس مسؤول عن عملية التدريب بالكامل
class Trainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=0.001, device='cuda', model_name="model"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name # عشان نحفظ الملف باسم مميز (ActionNet or HitNet)
        
        # إعدادات التدريب القياسية
        self.criterion = nn.CrossEntropyLoss() # دالة حساب الخطأ للتصنيف
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate) # المحسن
        
        # مكان حفظ الموديلات (Checkpoints)
        self.save_dir = "/content/drive/MyDrive/cnndataset/checkpoints"
        os.makedirs(self.save_dir, exist_ok=True)

    def train_one_epoch(self):
        self.model.train() # وضع التدريب
        running_loss = 0.0
        correct = 0
        total = 0
        
        # شريط التقدم
        loop = tqdm(self.train_loader, leave=False)
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # 1. تصفير التدرجات
            self.optimizer.zero_grad()
            
            # 2. التوقع (Forward Pass)
            outputs = self.model(inputs)
            
            # 3. حساب الخطأ (Loss)
            loss = self.criterion(outputs, labels)
            
            # 4. التعلم (Backward Pass)
            loss.backward()
            
            # 5. تحديث الأوزان
            self.optimizer.step()
            
            # حساب الإحصائيات
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # تحديث الشريط
            loop.set_description(f"Loss: {loss.item():.4f}")
            
        accuracy = 100 * correct / total
        return running_loss / len(self.train_loader), accuracy

    def validate(self):
        self.model.eval() # وضع الاختبار
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad(): # توفير الذاكرة
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = 100 * correct / total
        return running_loss / len(self.val_loader), accuracy

    def fit(self, epochs=10):
        print(f"🚀 Start Training {self.model_name}...")
        best_acc = 0.0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()
            
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # حفظ أفضل موديل فقط
            if val_acc >= best_acc:
                best_acc = val_acc
                save_path = os.path.join(self.save_dir, f"{self.model_name}_best.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"💾 Model Saved! (Best Acc: {best_acc:.2f}%)")
            
            print("-" * 30)
        
        print("✅ Training Complete!")
