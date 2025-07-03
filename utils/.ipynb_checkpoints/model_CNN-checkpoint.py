import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights

# 1. 先定義 channel_dict，讓 ConvNet 知道 cifar10 是 3 通道
channel_dict = {
    'cifar10': 3,
    'mnist':   1,
    # 如果還有其他 dataset，就繼續加上去
}

# 2. 定義 ConvNet（假設你已經把剛剛那段程式貼進來了）
class ConvNet(nn.Module):
    def __init__(self,
                 num_classes=10,
                 net_width=128,
                 net_depth=3,
                 net_act='relu',
                 net_norm='instancenorm',
                 net_pooling='avgpooling',
                 im_size=(32,32),
                 dataset='cifar10'):
        super(ConvNet, self).__init__()
        channel = channel_dict.get(dataset)              # 對應到 cifar10 → 3
        self.features, shape_feat = self._make_layers(
            channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size
        )
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        print(f"num feat {num_feat}")
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.get_feature(x)
        out = self.classifier(out)
        return out

    def get_feature(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit(f'unknown activation function: {net_act}')

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit(f'unknown net_pooling: {net_pooling}')

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = [C, H, W]
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit(f'unknown net_norm: {net_norm}')

    def _make_layers(self,
                     channel,
                     net_width,
                     net_depth,
                     net_norm,
                     net_act,
                     net_pooling,
                     im_size):
        layers = []
        in_channels = channel
        # 如果輸入是 28×28（例如 MNIST），就預設擴成 32×32
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    net_width,
                    kernel_size=3,
                    padding=3 if (channel == 1 and d == 0) else 1
                )
            )
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers.append(self._get_normlayer(net_norm, shape_feat))
            layers.append(self._get_activation(net_act))

            in_channels = net_width
            if net_pooling != 'none':
                layers.append(self._get_pooling(net_pooling))
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat

class SVHNNet(nn.Module):
    def __init__(self, num_classes=10, net_depth=3, net_width=128, net_norm='batchnorm', 
                 net_act='relu', net_pooling='maxpooling', dataset='svhn', im_size=(32, 32)):
        super(SVHNNet, self).__init__()
        channel = {'svhn': 3, 'cifar10': 3, 'mnist': 1}.get(dataset, 3)
        if im_size[0] == 28:
            im_size = (32, 32)
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, 
                                                     net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _get_activation(self, net_act):
        if net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(f'Unknown activation: {net_act}')

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError(f'Unknown pooling: {net_pooling}')

    def _get_normlayer(self, net_norm, shape_feat):
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0])
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0])
        else:
            return None

    def _make_layers(self, in_channels, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        shape_feat = [in_channels, im_size[0], im_size[1]]
        activation = self._get_activation(net_act)
        pool = self._get_pooling(net_pooling) if net_pooling != 'none' else None
        for d in range(net_depth):
            layers.append(nn.Conv2d(shape_feat[0], net_width, kernel_size=3, padding=1))
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers.append(self._get_normlayer(net_norm, shape_feat))
            layers.append(activation)
            if pool and d % 2 == 1:
                layers.append(pool)
                shape_feat[1] //= 2
                shape_feat[2] //= 2
        return nn.Sequential(*layers), shape_feat

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

# class SVHNNet(nn.Module):
#     def __init__(self, num_classes: int = 10):
#         super(SVHNNet, self).__init__()
#         # 5 個卷積層
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)   # Input: 3×32×32 → 64×32×32 :contentReference[oaicite:0]{index=0}
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # → 128×32×32
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)# → 256×32×32
#         self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)# → 256×32×32
#         self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)# → 512×32×32

#         # 池化層 (每 2 層後做一次 2×2 max pooling)
#         self.pool = nn.MaxPool2d(2, 2)

#         # 2 層全連接
#         # 經過兩次 pooling 後空間大小: 32→16→8 → feature map 大小 512×8×8
#         self.fc1 = nn.Linear(8192, 1024)  # 第一層全連接 :contentReference[oaicite:1]{index=1}
#         self.fc2 = nn.Linear(1024, num_classes) # 最後分類層

#         # Dropout 可強化泛化能力
#         self.dropout = nn.Dropout(p=0.5)

#     def forward(self, x):
#         # 卷積 + ReLU + Pool
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)            # → 128×16×16

#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = self.pool(x)            # → 256×8×8

#         x = F.relu(self.conv5(x))
#         x = self.pool(x)            # → 512×4×4 (若想要更深可再 pool)

#         # # 將特徵圖攤平成一維向量
#         # x = x.reshape(x.size(0), -1)
        
#         # 拉平
#         x = x.reshape(x.size(0), -1)

#         # 全連接 + Dropout + ReLU
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ImprovedSVHNNet(nn.Module):
#     def __init__(self, num_classes=10, net_depth=5, net_widths=[64, 128, 256, 256, 512], 
#                  net_norm='batchnorm', net_act='relu', net_pooling='maxpooling'):
#         super(ImprovedSVHNNet, self).__init__()
#         self.features, shape_feat = self._make_layers(net_widths, net_depth, net_norm, net_act, net_pooling)
#         num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
#         self.dropout = nn.Dropout(p=0.3)
#         self.fc1 = nn.Linear(num_feat, 1024)  # 中間全連接層
#         self.fc2 = nn.Linear(1024, num_classes)  # 分類層

#     def forward(self, x):
#         x = self.features(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return x

#     def _get_activation(self, net_act):
#         if net_act == 'relu':
#             return nn.ReLU(inplace=True)
#         elif net_act == 'leakyrelu':
#             return nn.LeakyReLU(negative_slope=0.01)
#         else:
#             raise ValueError(f'Unknown activation: {net_act}')

#     def _get_pooling(self, net_pooling):
#         if net_pooling == 'maxpooling':
#             return nn.MaxPool2d(kernel_size=2, stride=2)
#         elif net_pooling == 'avgpooling':
#             return nn.AvgPool2d(kernel_size=2, stride=2)
#         else:
#             raise ValueError(f'Unknown pooling: {net_pooling}')

#     def _get_normlayer(self, net_norm, out_channels):
#         if net_norm == 'batchnorm':
#             return nn.BatchNorm2d(out_channels)
#         elif net_norm == 'instancenorm':
#             return nn.GroupNorm(out_channels, out_channels)
#         else:
#             return None

#     def _make_layers(self, net_widths, net_depth, net_norm, net_act, net_pooling):
#         layers = []
#         in_channels = 3  # 固定為 SVHN 的 3 通道
#         shape_feat = [in_channels, 32, 32]  # 固定為 32×32
#         activation = self._get_activation(net_act)
#         pool = self._get_pooling(net_pooling) if net_pooling != 'none' else None
        
#         for d in range(net_depth):
#             out_channels = net_widths[d]
#             # 卷積層
#             layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
#             if net_norm != 'none':
#                 layers.append(self._get_normlayer(net_norm, out_channels))
#             layers.append(activation)
#             layers.append(nn.Dropout2d(p=0.2))  # 卷積層 Dropout
#             # 殞地連接（若通道數匹配）
#             if d % 2 == 0 and d < net_depth - 1 and net_widths[d] == net_widths[d + 1]:
#                 layers.append(ResidualConnection(net_widths[d]))
#             # 池化（僅在第 2、4 層後）
#             if pool and d in [1, 3]:
#                 layers.append(pool)
#                 shape_feat[1] //= 2
#                 shape_feat[2] //= 2
#             in_channels = out_channels
#             shape_feat[0] = out_channels
        
#         return nn.Sequential(*layers), shape_feat

# class ResidualConnection(nn.Module):
#     def __init__(self, channels):
#         super(ResidualConnection, self).__init__()
#         self.channels = channels

#     def forward(self, x):
#         identity = x
#         return identity  # 僅返回 identity，與後續層相加


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