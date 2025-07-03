from collections import OrderedDict
from typing import List, Tuple, Optional, Dict, Union, Callable
import matplotlib.pyplot as plt
from datasets.utils.logging import enable_progress_bar
enable_progress_bar()
import time
import threading
import random
import os
import logging
# 設定 Flower 和 Ray 的 log 級別，只顯示 CRITICAL 級別以上的訊息
logging.getLogger("flwr").setLevel(logging.CRITICAL)
logging.getLogger("ray").setLevel(logging.CRITICAL)
# os.environ["RAY_DEDUP_LOGS"] = "0"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from sklearn.cluster import AgglomerativeClustering

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.amp import autocast, GradScaler  # ✅ AMP 混合精度
torch.backends.cudnn.benchmark = True

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays , NDArrays, Context, Metrics, MetricsAggregationFn    
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, Strategy
from flwr.server.history import History
from flwr.simulation import run_simulation
from flwr.simulation import start_simulation
from flwr_datasets import FederatedDataset
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from koshino_FL.koshino_model_CNN import Net
from koshino_FL.koshino_train_v6 import test, lookahead_train, lookahead_UPA_train, lookahead_TPA_train
from koshino_FL.koshino_loaddata import load_datasets
from koshino_FL.koshino_others import get_parameters, set_parameters, evaluate_and_plot_confusion_matrix
from koshino_FL.koshino_Similarity_Measurement import cosine_similarity, pearson_correlation, compress_parameters, _compute_similarity_compressed
from koshino_FL.koshino_strategy import MyFedAvgWithDynamic
from koshino_FL.koshino_history import history
from koshino_FL.koshino_client import FlowerClient, AClient

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
# disable_progress_bar()

class DynamicClientManager(SimpleClientManager):
    def __init__(self):
        super().__init__()
        self.online_clients = set()
        self.offline_clients = set()  # 新增離線列表
        self.off_clients: dict[str, ClientProxy] = {}  # 正確初始化為空字典

    def register(self, client: ClientProxy) -> bool:
        if client.cid in self.offline_clients:
            self.offline_clients.remove(client.cid)
        self.online_clients.add(client.cid)
        print(f"[Status] Client {client.cid} 已上線 ✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓")
        return super().register(client)

    def unregister(self, client: ClientProxy) -> None:
        if client.cid in self.online_clients:
            self.off_clients[client.cid] = client
            self.online_clients.remove(client.cid)
            self.offline_clients.add(client.cid)  # 移至離線列表
            print(f"[Status] Client {client.cid} 已下線 ✗")
        super().unregister(client)

    def sample(self, num_clients: int, min_num_clients: Optional[int] = None) -> List[ClientProxy]:
        available = [cid for cid in self.online_clients if cid in self.clients]
        num_available = len(available)
        if num_available == 0:
            return []
        n_sample = max(num_clients, min_num_clients or 0)
        sampled_cids = random.sample(available, min(n_sample, num_available))
        return [self.clients[cid] for cid in sampled_cids]