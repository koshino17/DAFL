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
    def __init__(self, partition_id, net, trainloader, valloader, attack_type: str = None, 
                 max_attack_nums: int = 0, attack_mode: str = "fixed", attack_increase_rounds: int = 100):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = DEVICE
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)  # Adjusted lr
        self.base_attack_type = attack_type  # 基礎攻擊類型
        self.max_attack_nums = max_attack_nums  # 最大攻擊者數量
        self.attack_mode = attack_mode  # 攻擊模式: "fixed" 或 "progressive"
        self.attack_increase_rounds = attack_increase_rounds  # 漸進模式的增加輪數
        
    def get_current_attack_type(self, server_round: int) -> str:
        """根據當前輪次和攻擊模式決定是否為攻擊者"""
        if self.base_attack_type is None:
            return None
            
        if self.attack_mode == "fixed":
            # 固定模式：如果初始設為攻擊者就一直是攻擊者
            return self.base_attack_type
            
        elif self.attack_mode == "progressive":
            # 漸進模式：攻擊者數量隨輪次增加
            if server_round <= self.attack_increase_rounds:
                # 計算當前輪次應該有多少攻擊者
                current_attack_nums = max(1, int((server_round / self.attack_increase_rounds) * self.max_attack_nums))
            else:
                # 超過增加輪數後，維持最大攻擊者數量
                current_attack_nums = self.max_attack_nums
            
            # 如果此客戶端ID小於當前攻擊者數量，則成為攻擊者
            if int(self.partition_id) < current_attack_nums:
                return self.base_attack_type
            else:
                return None
        
        return None

    def get_parameters(self, config):
        return [param.cpu().numpy() for param in self.net.state_dict().values()]

    def get_parameters_dict(self, config):
        """Helper to return parameters as an OrderedDict for weights_substraction."""
        return OrderedDict([(name, param) for name, param in self.net.state_dict().items()])

    def fit(self, parameters, config):
        try:
            # 獲取當前輪次
            server_round = config.get("server_round", 1)
            
            # 動態決定當前是否為攻擊者
            current_attack_type = self.get_current_attack_type(server_round)
            
            # 如果是漸進模式，輸出攻擊者狀態變化
            # if self.attack_mode == "progressive":  # 只在前5輪輸出避免太多log
            #     attack_status = "攻擊者" if current_attack_type else "正常客戶端"
            #     if attack_status == "攻擊者":
            #         print(f"Round {server_round}, Client {self.partition_id}: {attack_status}")

            if current_attack_type == "UPA":
                grad, current_loss = lookahead_UPA_train(self.net, self.trainloader, parameters, config, self.partition_id, verbose=False)
            elif current_attack_type == "TPA":
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