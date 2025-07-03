from collections import OrderedDict
from typing import List, Tuple, Optional, Dict, Union, Callable
import matplotlib.pyplot as plt
import numpy as np
from datasets.utils.logging import enable_progress_bar
enable_progress_bar()
import time
import math
import os
import pandas as pd
from IPython.display import display
os.environ["RAY_DEDUP_LOGS"] = "0"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays , NDArrays, Context, Metrics, MetricsAggregationFn    
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr.simulation import start_simulation
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from koshino_FL.koshino_model_CNN import Net
# from koshino_FL.airbench94_muon import CifarNet as Net
# from koshino_FL.koshino_model_CNN import MobileNetV3_Small as Net
from koshino_FL.koshino_train_v6 import test
from koshino_FL.koshino_others import set_parameters
from koshino_FL.koshino_Similarity_Measurement import  _compute_similarity_compressed
from koshino_FL.koshino_loaddata import get_cached_datasets
from koshino_FL.koshino_history import history

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
# disable_progress_bar()


def check_and_clean_params(
    params: List[np.ndarray],
    cid: str
) -> List[np.ndarray]:
    """若參數含有 NaN/Inf，做一些簡單處理（例如直接設為0）."""
    cleaned_params = []
    for idx, w in enumerate(params):
        if np.isnan(w).any() or np.isinf(w).any():
            print(f"[Warning] Client {cid}, layer {idx} has NaN/Inf - replacing with zeros.")
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        cleaned_params.append(w)
    return cleaned_params

def parameters_to_vector(numpy_params: List[np.ndarray]) -> torch.Tensor:
    flat_list = []
    for p in numpy_params:
        flat_list.append(p.reshape(-1))  # 攤平
    merged = np.concatenate(flat_list, axis=0)
    return torch.from_numpy(merged)

def gradient_of(
    client_params: List[np.ndarray],
    global_params: List[np.ndarray]
) -> List[np.ndarray]:
    return [cp - gp for cp, gp in zip(client_params, global_params)]

def init_or_check_client_info_table(
    client_info_table: Dict[str, Dict[str, float]],
    client_ids: List[str],
) -> None:
    """
    檢查 client_info_table 是否已有對應的 client id，
    若沒有則初始化該 client 在表裡的紀錄。
    """
    for cid in client_ids:
        if cid not in client_info_table:
            # 初始化 reputation, last_loss 等等
            client_info_table[cid] = {
                "partition_id": -1,
                "client_accuracy": 0.0,
                "is_suspicious": False,
                "reputation": 0.3,        # 綜合聲譽（可改為長期聲譽）
                "short_term_rs": 0.3,     # 新增：短期RS
                "long_term_rs": 0.3,      # 新增：長期RS
                "client_loss": 9999.9,
                "loss_history": [],       # 新增：用於計算波動度
                "similarity": -1.0,
                "similarity_history": [],  # 新增：用於長期可靠性
                "shapley": 0.0,
                "shapley_history": [],
            }

def re_aggregate_without(
    client_params_dict: Dict[str, List[np.ndarray]],
    exclude_cid: str,
    client_info_table: Dict[str, Dict[str, float]],
    beta: float = 0.7,
) -> Optional[List[np.ndarray]]:
    """
    重新聚合(加權)所有 client 裡，排除 exclude_cid 的參數。
    回傳該「去掉某人」後的聚合模型參數。
    """
    # 1. 過濾掉想要排除的 client
    partial_clients = {
        cid: params for cid, params in client_params_dict.items() 
        if cid != exclude_cid
    }
    if not partial_clients:
        # 意味著只有這個 cid 一個客戶端在傳參數，移除後就空了
        return None
    
    # 2. 做 NaN/Inf 清洗
    for cid, params in partial_clients.items():
        partial_clients[cid] = check_and_clean_params(params, cid)
    
    # 3. 根據 short_term_rs/long_term_rs 做加權聚合
    weighted_params = None
    total_weight = 0.0
    for cid, params in partial_clients.items():
        info = client_info_table.get(cid, {})
        st_rs = info.get("short_term_rs", 0.0)
        lt_rs = info.get("long_term_rs", 0.0)
        
        combined_weight = beta * lt_rs + (1 - beta) * st_rs
        total_weight += combined_weight
        
        if weighted_params is None:
            weighted_params = [combined_weight * p for p in params]
        else:
            for idx, p in enumerate(params):
                weighted_params[idx] += combined_weight * p
    
    if total_weight <= 1e-9:
        # 若發現加權後幾乎為0，代表所有客戶端的聲譽都很低(或某些邏輯錯誤)
        print("[Warning] re_aggregate_without: total_weight=0 => fallback returns None")
        return None
    
    aggregated = [p / total_weight for p in weighted_params]
    return aggregated

def evaluate_model(params: List[np.ndarray], test_loader: DataLoader) -> float:
    net = Net().to(DEVICE)
    set_parameters(net, params)
    loss, _ = test(net, test_loader)
    return float(loss)

def calculate_shapley(
    global_model: List[np.ndarray],
    client_models: Dict[str, List[np.ndarray]],
    test_loader: DataLoader,
    client_info_table: Dict[str, Dict[str, float]],
    beta: float = 0.7
) -> Dict[str, float]:
    """
    以Leave-one-out的方式近似各客戶端的Shapley值:
      1) 先計算包含全部客戶端的聚合模型(也可直接用 global_model 參考值)
      2) 分別「移除cid」後重新聚合一次做 evaluate
      3) Shapley = |(no-cid-model-loss) - (base_loss)|
    """
    shapley_values = {}
    
    # 0) 準備 base_loss
    #    如果您確定 global_model 就是「包含所有客戶端聚合」後的結果，可直接用它
    base_loss = evaluate_model(global_model, test_loader)
    
    # 1) 準備一份「所有 client 參數」加權聚合 => fallback_base_model
    #    若您希望跟 global_model 一模一樣，可直接略過這一步，用 global_model 即可
    #    不過也可以用 "client_models" 做一次 sanitize。
    all_cleaned = {}
    for cid, p in client_models.items():
        all_cleaned[cid] = check_and_clean_params(p, cid)
    # 重新加權聚合
    all_aggregated = re_aggregate_without(all_cleaned, exclude_cid=None, 
                                          client_info_table=client_info_table, 
                                          beta=beta)
    if all_aggregated is None:
        # 表示可能只有1個client => fallback
        all_aggregated = global_model
    
    # 2) 用該全量聚合模型的 loss 當 base_loss (也可直接用 global_model 的 evaluate)
    fallback_base_loss = evaluate_model(all_aggregated, test_loader)
    
    # 3) 逐個 client 做「leave-one-out 聚合」再 evaluate
    for cid in client_models:
        # 跳過參數空的
        if not client_models[cid]:
            shapley_values[cid] = 0.0
            continue
        
        # (a) 重新聚合 (排除該cid)
        temp_aggregated = re_aggregate_without(all_cleaned, cid, client_info_table, beta=beta)
        if temp_aggregated is None:
            # 表示移除後沒有參與者 => shapley=base_loss - ??? => 可自行定義為0
            shapley_values[cid] = 0.0
            continue
        
        # (b) 評估 temp_loss
        temp_loss = evaluate_model(temp_aggregated, test_loader)
        
        # (c) Shapley = |(no-cid-loss) - (all-loss)|
        shap = abs(temp_loss - fallback_base_loss)
        shapley_values[cid] = shap
    
    return shapley_values

def update_client_info_table(
    client_info_table: Dict[str, Dict[str, float]],
    fit_results: List[Tuple[ClientProxy, FitRes]],
    global_model: List[np.ndarray],  # 新增參數
    alpha: float = 0.5,
) -> None:
    """
    根據本輪的訓練結果 (loss, metrics...) 更新 client_info_table。
    除了更新 loss 與 reputation，也更新 similarity（看作是慢速更新與全局參數間的相似性）。
    """
    # print("[INFO] global_model :", global_model)
    _, _, test_loader = get_cached_datasets(0)
    trained_cids = {client_proxy.cid for (client_proxy, _) in fit_results}
    # print('[INFO] trained_cids:', trained_cids)
    trained_partitions = {
        fit_res.metrics.get("partition_id", client_proxy.cid)
        for (client_proxy, fit_res) in fit_results
    }
    print('[INFO] trained_partitions:', trained_partitions)
    
    # (修正) 準備當前 client_models 給 Shapley 計算
    client_models_dict = {}
    for cp, fit_res in fit_results:
        cid = cp.cid
        
        # 清理 NaN/Inf
        raw_params = parameters_to_ndarrays(fit_res.parameters)
        raw_params = check_and_clean_params(raw_params, cid)
        client_models_dict[cid] = raw_params
    
    # (修正) 用 leave-one-out 方式計算 Shapley
    shapley_values = calculate_shapley(
        global_model=global_model,
        client_models=client_models_dict,
        test_loader=test_loader,
        client_info_table=client_info_table,
        beta=0.7  # 您自定義
    )
    # print("[INFO] shapley_values:", shapley_values)
    
    # 寫入 client_info_table
    for cid, shap_val in shapley_values.items():
        client_info_table[cid]["shapley"] = shap_val
        hist = client_info_table[cid].get("shapley_history", [])
        hist.append(shap_val)
        if len(hist) > 5:
            hist.pop(0)
        client_info_table[cid]["shapley_history"] = hist

    # 3) 更新其餘資訊 (loss, accuracy, similarity, short_term_rs...)
    for (client_proxy, fit_res) in fit_results:
        cid = client_proxy.cid
        fit_metrics = fit_res.metrics

        # (a) 基本資訊
        # client_info_table[cid]["partition_id"] = fit_metrics["partition_id"]
        maybe_pid = fit_metrics.get("partition_id", -1)
        if maybe_pid != -1:
            client_info_table[cid]["partition_id"] = maybe_pid
        latest_loss = float(fit_metrics.get("client_loss", 9999.9))
        latest_acc = float(fit_metrics.get("client_accuracy", 0.0))
        client_info_table[cid]["client_loss"] = latest_loss
        client_info_table[cid]["client_accuracy"] = latest_acc

        # (b) 維護 loss_history
        loss_history = client_info_table[cid].get("loss_history", [])
        if latest_loss < 9999.9:
            loss_history.append(latest_loss)
        if len(loss_history) > 5:
            loss_history.pop(0)
        client_info_table[cid]["loss_history"] = loss_history

        # (c) 維護 accuracy_history（新增）
        accuracy_history = client_info_table[cid].get("accuracy_history", [])
        accuracy_history.append(latest_acc)
        if len(accuracy_history) > 10:  # 假設長期要看最近10輪
            accuracy_history.pop(0)
        client_info_table[cid]["accuracy_history"] = accuracy_history

        # (d) 計算與 global params 的 similarity
        client_params_nd = parameters_to_ndarrays(fit_res.parameters)
        
        if global_model is not None:
            #計算similarity
            similarity = _compute_similarity_compressed(global_model, client_params_nd)
            # print(f"[INFO] Client {cid} similarity: {similarity:.4f}")
            client_info_table[cid]["similarity"] = similarity
            sim_history = client_info_table[cid].get("similarity_history", [])
            sim_history.append(similarity)
            if len(sim_history) > 5:
                sim_history.pop(0)
            client_info_table[cid]["similarity_history"] = sim_history

        # (e) 計算短期RS (與原程式相同)
        loss_std = np.std(loss_history) if len(loss_history) >= 2 else 0
        similarity_now = client_info_table[cid]["similarity"]

        short_term_rs = (
            0.6 * latest_acc +
            0.2 * (1 - loss_std) +
            0.2 * max(0, similarity_now)
        )
        client_info_table[cid]["short_term_rs"] = short_term_rs

    # === 4) 計算「步驟三：長期 RS (LongTermScore)」 ===
    #    - 長期表現 (Performance) = accuracy_history 的平均
    #    - 長期可靠度 (Reliability) = 1 - std(accuracy_history)（或其他定義）
    #    - 長期Shapley (Shapley) = shapley_history 的平均
    #    - 最後整合： LongTermScore = w3*Performance + w4*Reliability + w5*Shapley

    w3, w4, w5 = 0.4, 0.3, 0.3  # 權重可自行調整
    for cid in trained_cids:
        info = client_info_table[cid]
        # (a) 取得歷史
        acc_hist = info.get("accuracy_history", [])
        shap_hist = info.get("shapley_history", [])

        # (b) 長期表現 Performance = 平均 accuracy
        if len(acc_hist) > 0:
            longterm_performance = float(np.mean(acc_hist))
        else:
            longterm_performance = 0.0

        # (c) 長期可靠度 Reliability = 1 - std(accuracy)，避免負值則取 max(0, 1 - std)
        if len(acc_hist) >= 2:
            acc_std = float(np.std(acc_hist))
        else:
            acc_std = 0.0
        longterm_reliability = max(0.0, 1.0 - acc_std)

        # (d) 長期 Shapley = shapley_history 平均
        if len(shap_hist) > 0:
            longterm_shapley = float(np.mean(shap_hist))
        else:
            longterm_shapley = 0.0

        # (e) 結合成 LongTermScore
        long_term_score = (
            w3 * longterm_performance +
            w4 * longterm_reliability +
            w5 * longterm_shapley
        )
        client_info_table[cid]["long_term_rs"] = long_term_score

    # (E) **步驟四**：將短期 RS 與長期 RS 整合成最終 Reputation
    #    1) 避免互相壓制，可用單純加權平均：RS_raw = α * ST + (1-α) * LT
    #    2) 做 reliability tuning 或數值裁切
    #    3) 若差異過大，可做懲罰或跳過

    α = 0.5  # 您可自行決定權重
    for cid in trained_cids:
        st_rs = client_info_table[cid]["short_term_rs"]
        lt_rs = client_info_table[cid]["long_term_rs"]

        # (1) 加權
        rs_raw = α * st_rs + (1 - α) * lt_rs

        # (2) 可選：若短期與長期差異過大 => 懲罰或調整
        diff_ratio = abs(st_rs - lt_rs) / (lt_rs + 1e-8)
        if diff_ratio > 0.8:
            # 舉例：若差異 > 0.8，將最終值乘以 0.9 懲罰
            rs_raw = rs_raw * 0.9

        # (3) 數值裁切 (避免過大或小)
        rs_final = max(0.0, min(1.0, rs_raw))  # 介於 0~1

        # 存到 reputation
        client_info_table[cid]["reputation"] = rs_final

def sanitize_aggregation(client_params_list: Dict[str, List[np.ndarray]],
                         client_info_table: Dict[str, Dict[str, float]]) -> List[np.ndarray]:
    if not client_params_list:
        print("[Warning] No valid client parameters to aggregate. Returning None.")
        return None

    weighted_params = None
    total_weight = 0.0
    for cid, params in client_params_list.items():
        info = client_info_table.get(cid, {})
        rep_score = info.get("reputation", 1.0)  # 步驟四後的最終 reputation

        total_weight += rep_score
        if weighted_params is None:
            weighted_params = [rep_score * p for p in params]
        else:
            for idx, p in enumerate(params):
                weighted_params[idx] += rep_score * p

    if total_weight < 1e-9:
        return None

    aggregated_ndarrays = [p / total_weight for p in weighted_params]
    return aggregated_ndarrays

def apply_time_decay(client_info_table: Dict[str, Dict[str, float]], 
                     decay_factor: float = 0.95):
    """在每輪開始前對長期RS施加時間衰減"""
    for cid in client_info_table:
        lt_rs = client_info_table[cid].get("long_term_rs", 1.0)
        client_info_table[cid]["long_term_rs"] = decay_factor * lt_rs

def detect_anomalies(client_info_table: Dict[str, Dict[str, float]], threshold: float = 0.3):
    for cid, info in client_info_table.items():
        rep_score = info.get("reputation", 0.0)
        # 例如：若 reputation < 0.1，直接視為可疑
        if rep_score < threshold:
            info["is_suspicious"] = True
        else:
            info["is_suspicious"] = False

def median_aggregation(client_params_list: Dict[str, List[np.ndarray]]) -> List[np.ndarray]:
    """
    對每一層的參數做 element-wise median。
    回傳聚合後的參數（List[np.ndarray]），維度與單一 client 的 params 相同。
    """
    # 先把所有客戶端的參數整理成 list of list
    # shape: n_clients x n_layers x (param_shape...)
    all_params = list(client_params_list.values())
    n_clients = len(all_params)
    if n_clients == 0:
        return []
    
    n_layers = len(all_params[0])  # 假設所有 client 的 layer 數相同
    aggregated = []
    
    for layer_idx in range(n_layers):
        # 收集所有 client 的同一層參數
        layer_stack = np.array([all_params[c][layer_idx] for c in range(n_clients)])
        # layer_stack shape = (n_clients, ...) 例如 (n_clients, 128, 256)
        
        # 計算該層每個 element 的 median => axis=0
        layer_median = np.median(layer_stack, axis=0)
        aggregated.append(layer_median)
    
    return aggregated

def trimmed_mean_aggregation(
    client_params_list: Dict[str, List[np.ndarray]],
    trim_ratio: float = 0.1
) -> List[np.ndarray]:
    """
    對每一層的參數做 element-wise trimmed mean。
    trim_ratio=0.1 代表每個位置會刪除前10%和後10%的數值，再對剩餘80%做平均。
    """
    all_params = list(client_params_list.values())
    n_clients = len(all_params)
    if n_clients == 0:
        return []
    
    n_layers = len(all_params[0])
    aggregated = []
    
    # 計算要刪除的數量
    k = int(n_clients * trim_ratio)
    
    for layer_idx in range(n_layers):
        layer_stack = np.array([all_params[c][layer_idx] for c in range(n_clients)])
        # shape = (n_clients, ...)
        
        # 對每個 element 進行排序 => 需要先攤平或用 np.sort(..., axis=0)
        # 做法1：先 reshape => (n_clients, -1) => 排序 => 取中間 => reshape回原形
        shape_original = layer_stack[0].shape
        layer_2d = layer_stack.reshape(n_clients, -1)  # => (n_clients, n_params)
        # 針對每個 column (對應一個 element) 排序
        layer_2d_sorted = np.sort(layer_2d, axis=0)
        
        # 刪除前 k 與後 k
        # 注意若 n_clients < 2k => 會出錯，需事先檢查
        valid_slice = layer_2d_sorted[k : n_clients - k, :]  # shape => ((n_clients-2k), n_params)
        
        # 取平均
        trimmed_mean_1d = np.mean(valid_slice, axis=0)
        
        # reshape 回原本形狀
        layer_trimmed_mean = trimmed_mean_1d.reshape(shape_original)
        aggregated.append(layer_trimmed_mean)
    
    return aggregated

class MyFedAvgWithDynamic(FedAvg):
    """
    在 MyFedAvg 基礎上，加入『動態客戶端管理』的功能，並使用 DataFrame 儲存客戶端資訊。
    """
    def __init__(self,
                 fraction_fit,
                 alpha=0.5,
                 aggregation_mode="robust",  # 新增
                 robust_method="median",     # "median" / "trimmed_mean" / "krum" / ...
                 **kwargs):
        super().__init__(**kwargs)
        self.fraction_fit = fraction_fit  # 想要每回合選擇多少比例的客戶端
        self.alpha = alpha  # 用於滑動平均
        self.aggregation_mode = aggregation_mode
        self.robust_method = robust_method
        self.client_info_table: Dict[str, Dict[str, float]] = {}  # 記錄客戶端資訊
        self.client_info_df = pd.DataFrame()  # 存儲客戶端資訊的 DataFrame
        self.previous_global_params = None  # 用來保存上一輪分發給客戶端的全局參數
        self.metric_history: Dict[str, Dict[str, List[float]]] = {}

    # def initialize_parameters(
    #     self, client_manager: ClientManager
    # ) -> Optional[Parameters]:
    #     """Initialize global model parameters."""
    #     net = Net()
    #     ndarrays = get_parameters(net)
    #     return ndarrays_to_parameters(ndarrays)
    
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: flwr.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, flwr.common.FitIns]]:
        # 在每輪開始時先做時間衰減
        apply_time_decay(self.client_info_table, decay_factor=0.95)
        # 記錄當前回合發出的全局參數（用來做 similarity 比較）
        self.previous_global_params = parameters_to_ndarrays(parameters)
        old_fraction = self.fraction_fit  # 暫存舊值

        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        fit_ins = FitIns(parameters, config)
    
        all_clients = list(client_manager.all().values())
        if server_round <= 1:
            self.fraction_fit = 1.0
            fit_ins = super().configure_fit(server_round, parameters, client_manager)
            # 初始化表格資訊（若該client還沒有記錄）
            all_client_ids = [c[0].cid for c in fit_ins]
            init_or_check_client_info_table(self.client_info_table, all_client_ids)
            self.fraction_fit = old_fraction
            return fit_ins
        
        # 先依照 cid 排序，確保結果固定
        sorted_clients = sorted(all_clients,
                                key=lambda x: self.client_info_table[x.cid]["reputation"],  # 改用長期RS排序
                                reverse=True,)
        
        # 計算要選取的客戶端數量 (例如10個客戶端，fraction_fit=0.5，則選取 int(10*0.5) = 5 個)
        num_clients_to_select = int(len(sorted_clients) * self.fraction_fit)
        
        # 選取前面 num_clients_to_select 個客戶端
        selected_clients = sorted_clients[:num_clients_to_select]
        
        return [(client, fit_ins) for client in selected_clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        fit_results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
    
    	# 1) 收集所有 client 的參數
        client_params_list = {}

        for client_proxy, fit_res in fit_results:
            cid = client_proxy.cid
            client_params_nd = parameters_to_ndarrays(fit_res.parameters)
            client_params_list[cid] = client_params_nd

        # 2) 更新 client_info_table
        update_client_info_table(self.client_info_table, fit_results, self.previous_global_params, alpha=self.alpha)

        # 更新 DataFrame 並印出
        self.client_info_df = pd.DataFrame.from_dict(self.client_info_table, orient='index')
        # print(f"\n📊 [Round {server_round} Client Info Table] 📊")
        # display(self.client_info_df)
        
        # ===== 新增：記錄本輪各 client 的指標 =====
        for cid, info in self.client_info_table.items():
            partition_id = info.get("partition_id", cid)  # 用 partition_id 當作識別
            # 若該 partition_id 尚未在 metric_history 中，先初始化
            if partition_id not in self.metric_history:
                self.metric_history[partition_id] = {
                    "reputation": [],
                    "short_term_rs": [],
                    "long_term_rs": [],
                    "similarity": [],
                    "shapley": [],
                }
            # 將本輪的指標記錄下來（若該 client 本輪未參與，則 client_info_table 仍保留上一輪值）
            self.metric_history[partition_id]["reputation"].append(info.get("reputation", 0.0))
            self.metric_history[partition_id]["short_term_rs"].append(info.get("short_term_rs", 0.0))
            self.metric_history[partition_id]["long_term_rs"].append(info.get("long_term_rs", 0.0))
            self.metric_history[partition_id]["similarity"].append(info.get("similarity", 0.0))
            self.metric_history[partition_id]["shapley"].append(info.get("shapley", 0.0))
        # ==========================================

        # Debug: 確保 `client_info_table` 已更新
        print(f"[DEBUG] Updated client_info_table (Round {server_round}):")

        #異常檢測，會直接跳過異常客戶端(要改)
        # detect_anomalies(self.client_info_table, threshold=0.6)

        # 3) 依照 aggregation_mode 決定要用哪一種聚合方式
        if self.aggregation_mode == "robust":
            # (a) 進一步分支 robust_method: "median", "trimmed_mean", "krum", "rfa", ...
            if self.robust_method == "median":
                aggregated_ndarrays = median_aggregation(client_params_list)
            elif self.robust_method == "trimmed_mean":
                aggregated_ndarrays = trimmed_mean_aggregation(client_params_list, trim_ratio=0.1)
            else:
                # 其他 robust 方法...
                aggregated_ndarrays = median_aggregation(client_params_list)  # 預設當 median
	
        elif self.aggregation_mode == "score":
            # 用你先前的 Weighted/Score-based 方案
            # 可能也要先篩掉 is_suspicious 的 client
            sanitized_params = {
                cid: p for cid, p in client_params_list.items()
                if not self.client_info_table[cid]["is_suspicious"]
            }
            aggregated_ndarrays = sanitize_aggregation(sanitized_params, self.client_info_table)
	
        else:
            # 如果都不是 => fallback 用預設 FedAvg
            return super().aggregate_fit(server_round, fit_results, failures)

		# 4) 若 aggregated_ndarrays = None，fallback 用 super
        if aggregated_ndarrays is None:
            print("[Warning] aggregator returned None. Fallback to FedAvg.")
            return super().aggregate_fit(server_round, fit_results, failures)
	
        # 5) 更新 self.previous_global_params
        self.previous_global_params = aggregated_ndarrays
	
        # 6) 轉回成 Flower 需要的型態 (Parameters)
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # 7) 回傳
        return aggregated_parameters, {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        
        losses = []
        accuracies = []
        # 將每個 client 回傳的 metrics 記錄到 history
        for client_proxy, evaluate_res in results:
            cid = client_proxy.cid
            if cid not in self.client_info_table:
                init_or_check_client_info_table(self.client_info_table, [cid])

            metrics = evaluate_res.metrics  # e.g. {"loss": 0.12, "accuracy": 0.87, ...}
            # print(metrics)
            if "client_loss" in metrics:
                loss_val = metrics["client_loss"]
                losses.append(loss_val)
                self.client_info_table[cid]["client_loss"] = loss_val
                # 將這個 loss 記錄到分散式 (distributed) 的 loss
                history.add_loss_distributed(server_round, loss_val)
            if "client_accuracy" in metrics:
                acc_val = metrics["client_accuracy"]
                accuracies.append(acc_val)
                self.client_info_table[cid]["client_accuracy"] = acc_val
                # 將這個 accuracy 記錄到分散式 (distributed) 的 metrics
                history.add_metrics_distributed(server_round, {"accuracy": acc_val})
            
            loss_history = self.client_info_table[cid].get("loss_history", [])
            if(loss_val < 9999.9):
                loss_history.append(loss_val)
            if len(loss_history) > 5:  # 保留最近5輪數據
                loss_history.pop(0)
            self.client_info_table[cid]["loss_history"] = loss_history
        
        # 先呼叫父類別，取得預設的聚合結果(通常是平均loss)
        aggregated_loss = super().aggregate_evaluate(server_round, results, failures)
        aggregated_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        
        return aggregated_loss, {"accuracy": aggregated_accuracy}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
