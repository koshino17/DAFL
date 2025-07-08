from collections import OrderedDict
from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import math
import numpy as np
from typing import List, Tuple, Dict
from flwr.common import Metrics

def get_parameters(net) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray], device: torch.device = torch.device("cpu")):
    params_dict = zip(net.state_dict().keys(), parameters)

    # Debug: 打印 state_dict 的 keys 和 parameters 長度
    state_dict_keys = list(net.state_dict().keys())
    # print(f"Expected state_dict keys: {len(state_dict_keys)}, Received parameters: {len(parameters)}")

    # 如果長度不匹配，直接報錯
    if len(state_dict_keys) != len(parameters):
        raise ValueError(f"Parameter mismatch! Expected {len(state_dict_keys)}, but got {len(parameters)}")

    # 正常載入，過濾 "num_batches_tracked"，並指定設備
    state_dict = OrderedDict({
        k: torch.tensor(v, dtype=torch.float32, device=device)
        for k, v in params_dict if "num_batches_tracked" not in k
    })
    net.load_state_dict(state_dict, strict=False)

def evaluate_and_plot_confusion_matrix(net, test_loader, DEVICE):
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
    
def record_metrics(
    metric_history: Dict[str, Dict[str, List[float]]],
    client_info_table: Dict[str, Dict[str, float]],
) -> None:
    """
    將本輪各 client 的指標記錄到 metric_history 中。
    """
    for cid, info in client_info_table.items():
        partition_id = info.get("partition_id", cid)
        if partition_id not in metric_history:
            metric_history[partition_id] = {
                "reputation": [],
                "short_term_rs": [],
                "long_term_rs": [],
                "similarity": [],
                "shapley": [],
            }
        metric_history[partition_id]["reputation"].append(info.get("reputation", 0.0))
        metric_history[partition_id]["short_term_rs"].append(info.get("short_term_rs", 0.0))
        metric_history[partition_id]["long_term_rs"].append(info.get("long_term_rs", 0.0))
        metric_history[partition_id]["similarity"].append(info.get("similarity", 0.0))
        metric_history[partition_id]["shapley"].append(info.get("shapley", 0.0))
# --- end external function ---

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
                "previous_params": None,  # 新增：儲存上一輪參數
            }

def _l2_distance(params1: List[np.ndarray], params2: List[np.ndarray]) -> float:
    """
    計算兩個參數列表之間的 L2 距離。
    
    Args:
        params1: 第一組參數（每層是一個 NumPy 陣列）
        params2: 第二組參數（每層是一個 NumPy 陣列）
    
    Returns:
        float: 參數之間的 L2 距離
    """
    # 檢查層數是否一致
    if len(params1) != len(params2):
        print(f"[ERROR] 參數層數不匹配：params1 有 {len(params1)} 層，params2 有 {len(params2)} 層")
        return 0.0
    
    total_distance = 0.0
    for i, (p1, p2) in enumerate(zip(params1, params2)):
        # 確保輸入是 NumPy 陣列
        try:
            p1 = np.asarray(p1)
            p2 = np.asarray(p2)
        except Exception as e:
            print(f"[ERROR] 層 {i} 無法轉換為 NumPy 陣列：{str(e)}")
            return 0.0
        
        # 檢查形狀是否一致
        if p1.shape != p2.shape:
            print(f"[ERROR] 層 {i} 形狀不匹配：params1 形狀 {p1.shape}，params2 形狀 {p2.shape}")
            return 0.0
        
        # 計算 L2 距離
        try:
            diff = p1.flatten() - p2.flatten()
            layer_distance = np.sum(diff ** 2)
            total_distance += layer_distance
            # print(f"[DEBUG] 層 {i} L2 距離：{np.sqrt(layer_distance):.6f}, 形狀：{p1.shape}")
        except Exception as e:
            print(f"[ERROR] 層 {i} 計算 L2 距離失敗：{str(e)}")
            return 0.0
    
    # 檢查是否為 NaN 或 Inf
    distance = np.sqrt(total_distance)
    if np.isnan(distance) or np.isinf(distance):
        print(f"[WARNING] L2 距離為 NaN 或 Inf：{distance}")
        return 0.0
    
    # print(f"[DEBUG] 總 L2 距離：{distance:.6f}")
    return float(distance)

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

# @timed
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

