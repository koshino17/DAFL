import torch
import numpy as np
from typing import List, Tuple, Dict
from collections import OrderedDict
from math import log, isnan
from torch.utils.data import DataLoader
import torch.nn as nn

from flwr.client import NumPyClient

from utils.train_test import test, lookahead_UPA_train, lookahead_TPA_train, train, local_evaluate
from utils.others import get_parameters, set_parameters
from utils.time import timed
from utils.weights_utils import weights_substraction, norm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, valloader, attack_type: str = None):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = DEVICE
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)  # Adjusted lr
        self.attack_type = attack_type

    def get_parameters(self, config):
        return [param.cpu().numpy() for param in self.net.state_dict().values()]

    def get_parameters_dict(self, config):
        """Helper to return parameters as an OrderedDict for weights_substraction."""
        return OrderedDict([(name, param) for name, param in self.net.state_dict().items()])

    def fit(self, parameters, config):
        try:
            if self.attack_type == "UPA":
                grad, current_loss = lookahead_UPA_train(self.net, self.trainloader, parameters, config, self.partition_id, verbose=False)
            elif self.attack_type == "TPA":
                grad, current_loss = lookahead_TPA_train(self.net, self.trainloader, parameters, config, self.partition_id, verbose=False)
            else:
                grad, current_loss = train(self.net, self.trainloader, parameters, config, self.partition_id, verbose=False)
            
            # 計算確定性係數
            try:
                certainty, weight = cal_certainty(self.net, parameters, grad, self.trainloader, self.device, self.optimizer)
            except Exception as inner_e:
                import traceback
                print(f"客戶端 {self.partition_id} 計算更新時出錯: {inner_e}")
                print(traceback.format_exc())
                certainty = 1.0
                weight = len(self.trainloader.dataset)
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else None  # 清理 GPU 記憶體
            # 確保返回正確的三元組格式
            return self.get_parameters(config), weight, {"certainty": float(certainty), "partition_id": self.partition_id}
        
        except Exception as e:
            import traceback
            print(f"客戶端 {self.partition_id} 訓練錯誤: {e}")
            print(traceback.format_exc())
            # 出錯時仍返回正確格式
            return self.get_parameters(config), len(self.trainloader.dataset), {"certainty": 1.0, "partition_id": self.partition_id}

    def evaluate(self, parameters, config):
        try:
            avg_loss, accuracy = local_evaluate(parameters, self.net, self.device, self.valloader, self.partition_id)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None  # 清理 GPU 記憶體
            return float(avg_loss), len(self.valloader.dataset), {
                "client_accuracy": float(accuracy),
                "client_loss": float(avg_loss),
                "partition_id": self.partition_id
            }
        except Exception as e:
            print(f"客戶端 {self.partition_id} 評估錯誤: {e}")
            return 0.0, len(self.valloader.dataset), {
                "client_accuracy": 0.0,
                "partition_id": self.partition_id
            }

def cal_certainty(net, parameters, grad, trainloader, device, optimizer):
    # 獲取全局參數和本地更新後參數
    global_weights = OrderedDict()
    for name, param in zip(net.state_dict().keys(), parameters):
        global_weights[name] = torch.tensor(param, dtype=torch.float, device=device)
    
    local_weights = OrderedDict()
    for name, param in net.state_dict().items():
        local_weights[name] = param.detach().clone().float()
    
    # 計算更新 - 現在確保兩個參數都是 OrderedDict
    update = weights_substraction(global_weights, local_weights)
    
    # 計算範數
    update_norm = norm(update) if update else 1.0
    grad_norm = norm(grad) if grad else 1.0
    
    # 避免除以零
    if grad_norm > 0 and update_norm > 0:
        norm_factor = update_norm / grad_norm
        local_lr = optimizer.param_groups[0]['lr']
        if norm_factor > 0 and local_lr > 0:
            certainty = log(norm_factor / local_lr) + 1
            # 確保確定性合理
            certainty = 1.0 if isnan(certainty) else max(0.1, min(2.0, certainty))
        else:
            certainty = 1.0
    else:
        certainty = 1.0
    
    # 樣本數作為權重
    weight = len(trainloader.dataset)
    
    return certainty, weight



