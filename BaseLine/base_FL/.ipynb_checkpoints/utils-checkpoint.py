import torch
import torch.nn as nn

from collections import OrderedDict
from typing import List, Tuple, Optional
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_parameters(net) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)

    # Debug: 打印 state_dict 的 keys 和 parameters 長度
    state_dict_keys = list(net.state_dict().keys())
    # print(f"Expected state_dict keys: {len(state_dict_keys)}, Received parameters: {len(parameters)}")

    # 如果長度不匹配，直接報錯
    if len(state_dict_keys) != len(parameters):
        raise ValueError(f"Parameter mismatch! Expected {len(state_dict_keys)}, but got {len(parameters)}")

    # 正常載入
    state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict if "num_batches_tracked" not in k})
    net.load_state_dict(state_dict, strict=False)

def evaluate_and_plot_confusion_matrix(net, test_loader):
    net.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch["img"].to(DEVICE, memory_format=torch.channels_last), batch["label"].to(DEVICE)
            outputs = net(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 計算混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    
    # 繪製混淆矩陣
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()