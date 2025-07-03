import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights

# 自定義簡單 CNN 模型
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # 第一個卷積層：輸入 3 個通道，輸出 32 個通道，卷積核大小 3x3，padding=1 保持尺寸
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         # 第二個卷積層：從 32 個通道到 64 個通道
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         # 池化層，用來縮小特徵圖尺寸
#         self.pool = nn.MaxPool2d(2, 2)
#         # 第三個卷積層：從 64 個通道到 128 個通道
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         # 全連接層：因為 CIFAR-10 原始圖像是 32x32，
#         # 經過三次 2x2 池化後，尺寸會變成 32/2/2/2 = 4 (即 4x4)
#         # 故輸入尺寸為 128 * 4 * 4 = 2048，接到一個 256 維隱藏層，再到 10 維輸出（10 類）
#         self.fc1 = nn.Linear(128 * 4 * 4, 256)
#         self.fc2 = nn.Linear(256, 10)
        
#     def forward(self, x):
#         # 第一層卷積＋BatchNorm＋ReLU激活，再做池化：尺寸從 32x32 變成 16x16
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.pool(x)
#         # 第二層卷積＋BatchNorm＋ReLU，再做池化：尺寸從 16x16 變成 8x8
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool(x)
#         # 第三層卷積＋BatchNorm＋ReLU，再做池化：尺寸從 8x8 變成 4x4
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.pool(x)
#         # 將特徵圖攤平成一維向量
#         x = x.reshape(x.size(0), -1)
#         # 全連接層
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # 添加 Dropout 層
        self.dropout_conv = nn.Dropout(p=0.1)  # 卷積層後的 Dropout
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout_fc = nn.Dropout(p=0.1)  # 全連接層後的 Dropout
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout_conv(x)  # 在卷積層後、池化前添加 Dropout
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)  # 在全連接層之間添加 Dropout
        x = self.fc2(x)
        return x

# MobileNetV3-Small
class MobileNetV3_Small(nn.Module):
    def __init__(self):
        super(MobileNetV3_Small, self).__init__()
        self.mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.mobilenet.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # CIFAR-10適配
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(576, 1280),  # 保持 MobileNetV3-Small 的結構
            nn.Hardswish(),
            # nn.Dropout(0.3),  # 添加 Dropout
            nn.Linear(1280, 10)  # 替換為 CIFAR-10 10 類別
        )

    def forward(self, x):
        return self.mobilenet(x)

class CIFARResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFARResNet18, self).__init__()
        # 使用 torchvision 提供的 ResNet18
        self.model = resnet18(num_classes=num_classes)
        # 調整第一層卷積：原本 kernel_size=7, stride=2, padding=3 適用於大尺寸圖像
        # 將其改為 kernel_size=3, stride=1, padding=1，並移除第一層最大池化
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        
    def forward(self, x):
        return self.model(x)

class CIFARResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFARResNet50, self).__init__()
        # 1) 載入預設的 ResNet-50，並設定輸出類別數
        self.model = resnet50(num_classes=num_classes)
        
        # 2) 調整首層卷積：kernel_size 3, stride 1, padding 1，去掉 maxpool
        self.model.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        self.model.maxpool = nn.Identity()  # 移除第一層 max pooling

        # 3) （可選）在最後一層前加 Dropout 增強正則化
        #    將全連接 fc 層改包成 Dropout → Linear
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=num_classes)
        )

    def forward(self, x):
        return self.model(x)

class SVHNNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(SVHNNet, self).__init__()
        # 5 個卷積層
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)   # Input: 3×32×32 → 64×32×32 :contentReference[oaicite:0]{index=0}
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # → 128×32×32
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)# → 256×32×32
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)# → 256×32×32
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)# → 512×32×32

        # 池化層 (每 2 層後做一次 2×2 max pooling)
        self.pool = nn.MaxPool2d(2, 2)

        # 2 層全連接
        # 經過兩次 pooling 後空間大小: 32→16→8 → feature map 大小 512×8×8
        self.fc1 = nn.Linear(8192, 1024)  # 第一層全連接 :contentReference[oaicite:1]{index=1}
        self.fc2 = nn.Linear(1024, num_classes) # 最後分類層

        # Dropout 可強化泛化能力
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 卷積 + ReLU + Pool
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)            # → 128×16×16

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)            # → 256×8×8

        x = F.relu(self.conv5(x))
        x = self.pool(x)            # → 512×4×4 (若想要更深可再 pool)

        # # 將特徵圖攤平成一維向量
        # x = x.reshape(x.size(0), -1)
        
        # 拉平
        x = x.reshape(x.size(0), -1)

        # 全連接 + Dropout + ReLU
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class CIFAREfficientNetV2S(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        # 1) 載入 EfficientNetV2-S，並設定是否使用 ImageNet 預訓練權重
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_v2_s(weights=weights)  # :contentReference[oaicite:0]{index=0}

        # 2) 修改第一層 conv，以適配 CIFAR-10 32×32 小圖 (若您習慣將圖放大到 224×224，也可跳過此步)
        #    原始為 Conv2d(3, 24, kernel_size=3, stride=2, padding=1)
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=3,
            out_channels=24,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # 3) 將分類頭從 1000 類改為 10 類
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),               # 原論文使用 0.2 Dropout
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 當 batch 大小較大時，可考慮使用 channels_last 加速
        x = x.to(memory_format=torch.channels_last)
        return self.backbone(x)

class CIFAREfficientNetV2M(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        # 載入 EfficientNetV2-M，選擇是否使用預訓練權重
        weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_v2_m(weights=weights)

        # 調整首層卷積以適配 32×32 輸入
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=3,
            out_channels=24,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # 修改分類頭：Dropout 後輸出 10 類
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(memory_format=torch.channels_last)
        return self.backbone(x)
    
class CIFAREfficientNetV2L(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        # 載入 EfficientNetV2-L，選擇是否使用預訓練權重
        weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_v2_l(weights=weights)

        # 調整首層卷積以適配小圖，並移除首階段 pooling
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.backbone.features[0][1] = nn.Identity()

        # 修改分類頭：Dropout 後輸出 10 類
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(memory_format=torch.channels_last)
        return self.backbone(x)