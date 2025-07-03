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
    'fashion_mnist': 1,
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

