import torch.nn.functional as F
import torch.nn as nn

# 自定義簡單 CNN 模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一個卷積層：輸入 3 個通道，輸出 32 個通道，卷積核大小 3x3，padding=1 保持尺寸
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # 第二個卷積層：從 32 個通道到 64 個通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 池化層，用來縮小特徵圖尺寸
        self.pool = nn.MaxPool2d(2, 2)
        # 第三個卷積層：從 64 個通道到 128 個通道
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # 全連接層：因為 CIFAR-10 原始圖像是 32x32，
        # 經過三次 2x2 池化後，尺寸會變成 32/2/2/2 = 4 (即 4x4)
        # 故輸入尺寸為 128 * 4 * 4 = 2048，接到一個 256 維隱藏層，再到 10 維輸出（10 類）
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        # 第一層卷積＋BatchNorm＋ReLU激活，再做池化：尺寸從 32x32 變成 16x16
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        # 第二層卷積＋BatchNorm＋ReLU，再做池化：尺寸從 16x16 變成 8x8
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        # 第三層卷積＋BatchNorm＋ReLU，再做池化：尺寸從 8x8 變成 4x4
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        # 將特徵圖攤平成一維向量
        x = x.reshape(x.size(0), -1)
        # 全連接層
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
