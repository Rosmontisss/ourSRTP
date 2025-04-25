import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchaudio
from sklearn.model_selection import train_test_split
from torchaudio.transforms import MelSpectrogram
from multiprocessing import Pool
import hashlib
import argparse
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
# 现有导入语句
import matplotlib.pyplot as plt
import random

from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
# ========== 可配置参数 ==========
CONFIG = {
    # 训练参数
    "BATCH_SIZE": 32,
    "EPOCHS": 1,
    "LEARNING_RATE": 1e-5,
    "TEST_SIZE": 0.3,
    
    # 调试参数
    "DEBUG_MODE": True,
    "REPORT_FREQ": 1,
    "SHOW_SAMPLE": True,
    "SHOW_DEVICE": True,
    
    # 数据参数
    "DATA_DIR": r'I:\抑郁症诊疗模型\depMam\new\data\audio',
    "AUGMENT_PROB": 0.5,
    "CACHE_ENABLED": False,
    "PLOT_LOSS": True,
    "SHOW_AUGMENTED": False,
    "MAX_LENGTH": 500,
    # 新增评估参数
    "PLOT_CM": True,
    "PLOT_AUC": True,
    "CLASS_NAMES": ["Non-Depressed", "Depressed"]
}
# ==============================

class DepressionAudioModel(nn.Module):
    def __init__(self, input_dim=128, time_dim=500):
        super().__init__()
        self.feature_dim = None  # 先定义属性
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 添加Dropout
            nn.Dropout(0.3),  # 添加Dropout
        )
        self._calculate_feature_dim(input_dim, time_dim)  # 然后计算
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # 增加Dropout比例
            nn.Linear(256, 1)
        )
        
    def _calculate_feature_dim(self, input_dim, time_dim):
        with torch.no_grad():
            x = torch.zeros(1, 1, input_dim, time_dim)
            x = self.feature_extractor(x)
            self.feature_dim = x.view(1, -1).size(1)  # 更安全的计算方式
            expected_dim = 32 * (input_dim//4) * (time_dim//4)
            if self.feature_dim != expected_dim:
                raise ValueError(f"特征维度不匹配: {self.feature_dim} vs {expected_dim}")

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class AudioDepressionDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.samples = []
        self.labels = []
        self.augment = augment
        self.sr = 16000
        self.max_length = CONFIG["MAX_LENGTH"]
        self.mel_transform = MelSpectrogram(
            sample_rate=self.sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        
        # 加载噪声样本
        self.noise_samples = []
        if augment:
            noise_dir = os.path.join(data_dir, 'background_noise')
            if os.path.exists(noise_dir):
                for fname in os.listdir(noise_dir):
                    if fname.endswith('.wav'):
                        noise, _ = torchaudio.load(os.path.join(noise_dir, fname))
                        self.noise_samples.append(noise)

        # 加载数据
        with Pool(4) as p:
            results = p.map(self._scan_dir, [
                (data_dir, 'depressed', 1),
                (data_dir, 'non_depressed', 0)
            ])
        for samples, labels in results:
            self.samples.extend(samples)
            self.labels.extend(labels)

    def _scan_dir(self, args):
        data_dir, cls, label = args
        samples = []
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"警告: 目录不存在 - {cls_dir}")  # 添加调试信息
            return [], []
            
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith('.wav'):  # 根据用户确认只检查.wav文件
                full_path = os.path.join(cls_dir, fname)
                if os.path.isfile(full_path):  # 确保是文件
                    samples.append(full_path)
                else:
                    print(f"警告: 不是有效文件 - {full_path}")
        print(f"加载到 {len(samples)} 个{cls}样本")  # 显示加载数量
        return samples, [label]*len(samples)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.samples[idx])
        waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        
        # 转换为单声道
        if waveform.shape[0] > 1:  # 如果是多声道
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # 取平均值转为单声道
        
        # 数据增强
        if self.augment and torch.rand(1).item() < CONFIG["AUGMENT_PROB"]:
            waveform = self._augment(waveform)
        
        # 生成Mel频谱
        mel = self.mel_transform(waveform)
        log_mel = torchaudio.functional.amplitude_to_DB(
            mel, 
            amin=1e-10,
            multiplier=10.0,  # 新增参数
            db_multiplier=1.0  # 新增参数
        )
        
        # 标准化
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        
        # 填充/截断
        if log_mel.shape[2] > self.max_length:
            log_mel = log_mel[:, :, :self.max_length]
        else:
            pad_size = self.max_length - log_mel.shape[2]
            log_mel = nn.functional.pad(log_mel, (0, pad_size))
            
        return log_mel, torch.tensor(self.labels[idx]).float()

    def _augment(self, waveform):
        # 添加时间扭曲和音量变化
        waveform = torchaudio.functional.speed(waveform, orig_freq=self.sr, factor=random.uniform(0.9, 1.1))
        waveform = waveform * random.uniform(0.8, 1.2)  # 音量变化
        return waveform

    def __len__(self):
        return len(self.samples)

# 修改训练函数中的保存逻辑
def train_model(data_dir, batch_size=32, epochs=10, lr=1e-4):
    print(f"正在使用数据目录: {os.path.abspath(data_dir)}")  # 显示实际使用的路径
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DepressionAudioModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # 添加weight_decay
    
    # 加载数据集
    full_dataset = AudioDepressionDataset(data_dir)
    train_idx, test_idx = train_test_split(
        range(len(full_dataset)),
        test_size=CONFIG["TEST_SIZE"],
        stratify=full_dataset.labels,
        random_state=42
    )
    
    # 创建训练集副本
    train_set = Subset(full_dataset, train_idx)
    test_set = Subset(full_dataset, test_idx)  # 确保测试集独立
    train_set.dataset = copy.deepcopy(train_set.dataset)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        batch_size=batch_size,
        num_workers=4
    )
    
    best_acc = 0.0
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 确保labels维度匹配outputs
            labels = labels.view(-1, 1)  # 将labels从[batch_size]变为[batch_size, 1]
            loss = criterion(outputs, labels)  # 移除squeeze()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                labels = labels.view(-1, 1)  # 同样修改验证部分的labels维度
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
        
        # 记录损失
        train_loss = epoch_loss / len(train_set)
        val_loss = val_loss / len(test_loader.dataset)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_correct/len(test_loader.dataset):.2%}")
        
    # 新增评估可视化部分
    # 替换原来的保存代码
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
    }, 'best_audio_model.pth')  # 保存完整训练状态
    
    # 修改加载方式
    if CONFIG["PLOT_CM"] or CONFIG["PLOT_AUC"]:
        # 修改模型加载方式
        checkpoint = torch.load('best_audio_model.pth', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.extend(probs.squeeze())
                all_labels.extend(labels.cpu().numpy())
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds)

        # 绘制混淆矩阵
        if CONFIG["PLOT_CM"]:
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=CONFIG["CLASS_NAMES"],
                       yticklabels=CONFIG["CLASS_NAMES"])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.show()

        # 绘制AUC曲线
        if CONFIG["PLOT_AUC"]:
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8,6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()

if __name__ == "__main__":
    # 移除原有的参数解析代码
    train_model(
        data_dir=CONFIG["DATA_DIR"],  # 使用配置文件中的路径
        batch_size=CONFIG["BATCH_SIZE"],
        epochs=CONFIG["EPOCHS"],
        lr=CONFIG["LEARNING_RATE"]
    )
