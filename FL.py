# from collections import OrderedDict
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt
from datasets.utils.logging import enable_progress_bar
enable_progress_bar()
import time
import threading
import random
import os
import torch


import flwr
from flwr.client import Client, ClientApp
from flwr.common import ndarrays_to_parameters, Context, Metrics
from flwr.server import ServerApp, ServerConfig, ServerAppComponents

from flwr.simulation import run_simulation

from utils.model_CNN import Net
from utils.model_CNN import SVHNNet
from utils.model_CNN import ConvNet
from utils.train_test import test
from utils.loaddata import get_cached_datasets
from utils.others import get_parameters, set_parameters, evaluate_and_plot_confusion_matrix


from utils.history import history
from utils.client import FlowerClient
from utils.clientmanger import DynamicClientManager
from utils.HRFA_strategy import HRFA
from utils.other_strategy import AdaFedAdamStrategy, MyFedAvg
from utils.dynamic_controll import background_online_offline_simulator


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
# disable_progress_bar()

NUM_CLIENTS = 100
NUM_ROUNDS = 500
NUM_EPOCHS = 5
ATTACK_TYPE = "UPA"
DFL = False
STRATEGY = "HRFA"  # 可選 "FedAvg", "AdaFedAdam", "HRFA"


DATASET = "cifar10" #"mnist", "cifar10", "svhn", "fashion_mnist"

#dynamic experiment
if DATASET == "cifar10":
    FRACTION = 0.6
    Q = 0.7
    ATTACK_NUMS = NUM_CLIENTS*0.28
    # ATTACK_NUMS = 0
    Net = ConvNet
elif DATASET == "fashion_mnist":
    FRACTION = 0.5
    Q = 0.8
    ATTACK_NUMS = NUM_CLIENTS*0.28
    # ATTACK_NUMS = 0
    Net = ConvNet
#static experiment
elif DATASET == "mnist":
    FRACTION = 0.5
    Q = 1.0
    ATTACK_NUMS = (NUM_CLIENTS//2)-1
    # ATTACK_NUMS = 0
    Net = Net
else :
    FRACTION = 1.0
    Q = 0.9
    ATTACK_NUMS = (NUM_CLIENTS//2)-1
    # ATTACK_NUMS = 0
    Net = SVHNNet


current_server_round = 0
new_round_event = threading.Event()

def get_current_round() -> int:
    return current_server_round

def client_fn(context: Context) -> Client:
    net = Net().to(DEVICE, memory_format=torch.channels_last)
	# Read the node_config to fetch data partition associated to this node
    partition_id = str(context.node_config["partition-id"])  # 強制轉換為字串
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader, _ = get_cached_datasets(partition_id, dataset_name=DATASET, num_partitions=NUM_CLIENTS, q=Q)

    if int(partition_id) < ATTACK_NUMS:
        return FlowerClient(partition_id, net, trainloader, valloader, ATTACK_TYPE).to_client()
    else:
        return FlowerClient(partition_id, net, trainloader, valloader, None).to_client()
# Create the ClientApp
client = ClientApp(client_fn=client_fn)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = []
    examples = []
    for num_examples, m in metrics:
        if "client_accuracy" not in m:
            print(f"Warning: Missing 'client_accuracy' in metrics for client")
            continue
        accuracies.append(num_examples * m["client_accuracy"])
        examples.append(num_examples)
    total_examples = sum(examples)
    if total_examples <= 0:
        print("Warning: No valid examples for aggregation. Returning 0.0")
        return {"accuracy": 0.0}
    return {"accuracy": sum(accuracies) / total_examples}


_, _, testloader = get_cached_datasets(0, dataset_name=DATASET, num_partitions=NUM_CLIENTS, q=Q)

def server_evaluate(server_round, parameters, config):
    global current_server_round
    # 更新全域變數，讓背景執行緒知道目前是第幾個 round
    current_server_round = server_round
    """Evaluate the global model after each round (不再畫 confusion matrix)."""
    start_time = time.time()  # 記錄開始時間
    net = Net().to(DEVICE)
    set_parameters(net, parameters)

    # 測試
    loss, accuracy = test(net, testloader)
    
    end_time = time.time()  # 記錄結束時間
    round_time = end_time - start_time  # 計算 round 時間
    
    history.add_loss_centralized(server_round, loss)
    history.add_metrics_centralized(server_round, {"accuracy": accuracy})

    # 只記錄最終模型，不畫 confusion matrix
    if server_round == NUM_ROUNDS:  # 最後一輪才返回模型
        torch.save(net, "SVHN.pth")
        # evaluate_and_plot_confusion_matrix(net, testloader, DEVICE)
        return loss, {"accuracy": accuracy}
    new_round_event.set()
    return loss, {"accuracy": accuracy}

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        # "local_epochs": 1 if server_round < 2 else NUM_EPOCHS,
        "local_epochs": NUM_EPOCHS,
        "train_mode": "lookahead",
    }
    return config

params = get_parameters(Net())
param_count = sum(p.numel() for p in Net().parameters() if p.requires_grad)
print(f"Trainable Parameters: {param_count:,d}")

if STRATEGY == "AdaFedAdam":
    print("Using AdaFedAdam strategy")
    strategy = AdaFedAdamStrategy(
        fraction_fit=FRACTION,
        fraction_evaluate=FRACTION,
        
        min_fit_clients=int(NUM_CLIENTS * FRACTION),  # 確保是整數
        min_evaluate_clients=int(NUM_CLIENTS * FRACTION),  # 確保是整數
        min_available_clients=int(NUM_CLIENTS * FRACTION),  # 確保是整數

        initial_parameters=ndarrays_to_parameters(params),
        evaluate_fn=server_evaluate,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
        net=Net().to(DEVICE),
    )
elif STRATEGY == "FedAvg":
    print("Using FedAvg strategy")
    strategy = MyFedAvg(
        fraction_fit=FRACTION,
        fraction_evaluate=FRACTION,
        
        min_fit_clients=int(NUM_CLIENTS * FRACTION),  # 確保是整數
        min_evaluate_clients=int(NUM_CLIENTS * FRACTION),  # 確保是整數
        min_available_clients=int(NUM_CLIENTS * FRACTION),  # 確保是整數

        initial_parameters=ndarrays_to_parameters(params),
        evaluate_fn=server_evaluate,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
    )
else:
    print("Using HRFA strategy")
    strategy = HRFA(
        fraction_fit=FRACTION,  # Sample 100% of available clients for training
        fraction_evaluate=FRACTION,  # Sample 50% of available clients for evaluation
        
        min_fit_clients=2,  # Never sample less than 10 clients for training
        min_evaluate_clients=2,  # Never sample less than 5 clients for evaluation
        min_available_clients=2,  # Wait until all 10 clients are available
        
        initial_parameters=ndarrays_to_parameters(params),  # Pass initial model parameters
        evaluate_fn=server_evaluate,  # 設定 evaluate_fn
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        on_fit_config_fn=fit_config,  # Pass the fit_config function
        on_evaluate_config_fn=fit_config,
        net=Net().to(DEVICE),
        # testloader = testloader,
    )


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""
    global testloader

    # 設定 ServerConfig，如同你的程式碼
    config = ServerConfig(num_rounds=NUM_ROUNDS)

    # 建立動態管理器
    client_manager = DynamicClientManager()
        
    # 建立並啟動背景執行緒，模擬客戶端動態上/下線
    simulator_thread = threading.Thread(
        target=background_online_offline_simulator,
        args=(client_manager, get_current_round, new_round_event, 0.5, 0.5, DFL),  # interval=30秒, toggle_rate=0.3
        daemon=True  # 設 daemon=True 可以在主程式結束時自動退出
    )
    simulator_thread.start()

    return ServerAppComponents(
        strategy=strategy,
        config=config,
        client_manager=client_manager,
    )


# Create the ServerApp
server = ServerApp(server_fn=server_fn)

# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign an entire GPU for each client
if DEVICE == "cuda":
    backend_config = {"client_resources": {"num_cpus": 2 , 
                                           "num_gpus": 0.2 #if (NUM_CLIENTS*FRACTION)>40 else 2/(NUM_CLIENTS*FRACTION)
                                          }}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`


# 讓 Flower 運行完整的 FL 訓練
start_time = time.time()  # 記錄開始時間

run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_CLIENTS,
    backend_config=backend_config,
)

end_time = time.time()  # 記錄結束時間
total_time = end_time - start_time  # 計算總時間

print(f"Total Training Time: {total_time:.2f} seconds")  # 顯示總時間



# history.metrics_centralized
best_epoch, best_accuracy = max(history.metrics_centralized['accuracy'], key=lambda x: x[1])

print(f"Best Epoch: {best_epoch}, Accuracy: {best_accuracy:.4f}")

import numpy as np

# 1. 從 history.losses_distributed 中整理出：{ round_num: [loss_client1, loss_client2, ...], ... }
accuracies_by_round = {}
for round_num, pid, loss in history.metrics_distributed.get('accuracy', []):
    if round_num not in accuracies_by_round:
        accuracies_by_round[round_num] = []
    accuracies_by_round[round_num].append(loss)

# 2. 按 round 排序，計算每一輪的 client_loss 標準差（母體標準差 ddof=0）
rounds = sorted(accuracies_by_round.keys())
fairness_std_per_round = []
for r in rounds:
    client_accuracies_this_round = accuracies_by_round[r]
    if len(client_accuracies_this_round) > 1:
        std_val = float(np.std(client_accuracies_this_round, ddof=0))
    else:
        std_val = 0.0  # 如果只有一個 client 或無資料，就設為 0
    fairness_std_per_round.append(std_val)


avg_fairness_std = float(np.mean(fairness_std_per_round))
print("500 輪公平性標準差平均值：", avg_fairness_std)
