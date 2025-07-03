from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
from flwr.common import EvaluateRes
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from collections import OrderedDict
from scipy.spatial.distance import cdist
from utils.history import history
from utils.weights_utils import weights_substraction
from utils.optim import AdaAdam

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def median_aggregation(params_dict: Dict[str, List[np.ndarray]]) -> List[np.ndarray]:
    num_layers = len(list(params_dict.values())[0])
    agg_params = []
    for layer_idx in range(num_layers):
        layer_params = np.array([params_dict[cid][layer_idx].flatten() for cid in params_dict])
        agg_params.append(np.median(layer_params, axis=0).reshape(list(params_dict.values())[0][layer_idx].shape))
    return agg_params

def non_IID_aggregation(model, optimizer, results, aggregated_params, previous_weights):
    """
    針對非IID數據進行參數聚合，結合客戶端確定性和權重
    
    Args:
        model: 模型
        optimizer: 優化器
        results: 客戶端訓練結果列表
        aggregated_params: 已聚合的參數
        previous_weights: 前一輪的權重
        
    Returns:
        更新後的參數（已轉換為numpy數組的列表）
    """
    # 驗證聚合參數
    expected_count = len(model.state_dict().keys())
    if len(aggregated_params) != expected_count:
        print(f"警告: 參數數量不匹配! 期望 {expected_count}, 得到 {len(aggregated_params)}")
        fresh_model = model
        fresh_params = [param.cpu().numpy() for param in fresh_model.state_dict().values()]
        return fresh_params
    
    # 將聚合參數載入模型
    device = next(model.parameters()).device
    state_dict = OrderedDict()
    for (name, _), nd in zip(model.state_dict().items(), aggregated_params):
        state_dict[name] = torch.tensor(nd, dtype=torch.float, device=device)
    model.load_state_dict(state_dict)
    
    # 更新當前參數
    current_params = {name: param.detach().clone().float() 
                      for name, param in model.named_parameters()}
    
    # 如果前一次權重為空，則初始化
    if previous_weights is None:
        previous_weights = current_params
        return aggregated_params
    
    # 從客戶端聚合更新和確定性
    total_weight = 0
    certainty_sum = 0
    pseudo_grad = OrderedDict()
    
    for client_proxy, fit_res in results:
        # 確認客戶端返回了確定性指標
        if "certainty" not in fit_res.metrics:
            print(f"警告: 客戶端 {client_proxy.cid} 未返回certainty，跳過該客戶端")
            continue
        
        # 轉換客戶端參數
        params = parameters_to_ndarrays(fit_res.parameters)
        client_params = OrderedDict()
        for (name, _), param in zip(model.state_dict().items(), params):
            client_params[name] = torch.tensor(param, device=device)
        
        # 計算更新（前一次與當前客戶端參數的差異）
        update = weights_substraction(previous_weights, client_params)
        
        # 使用客戶端提供的確定性和權重
        weight = fit_res.num_examples
        certainty = fit_res.metrics["certainty"]
        
        total_weight += weight
        certainty_sum += certainty * weight
        
        # 累積偽梯度
        for key in update:
            if key not in pseudo_grad:
                pseudo_grad[key] = weight * update[key]
            else:
                pseudo_grad[key] += weight * update[key]
    
    # 如果總權重為零，則無法聚合
    if total_weight <= 0:
        print(f"警告: 總權重為0，無法聚合")
        return aggregated_params
    
    # 計算平均確定性和偽梯度
    certainty = certainty_sum / total_weight
    for key in pseudo_grad:
        pseudo_grad[key] = pseudo_grad[key] / total_weight
    
    # 將偽梯度應用到優化器
    optimizer.zero_grad()
    for k, w in model.named_parameters():
        if w.requires_grad and k in pseudo_grad:
            w.grad = pseudo_grad[k]
    
    # 使用優化器更新模型
    optimizer.set_confidence(certainty)
    optimizer.step()
    
    # 準備最終參數（確保是numpy數組，而不是包含其他Python對象）
    final_state_dict = model.state_dict()
    updated_params = [param.cpu().detach().numpy() for param in final_state_dict.values()]
    
    return updated_params

#爛
# aggregated = trimmed_mean_aggregation(filtered_params)
def trimmed_mean_aggregation(params_dict: Dict[str, List[np.ndarray]], trim_percent: float = 0.15) -> List[np.ndarray]:
    num_clients = len(params_dict)
    num_trim = int(num_clients * trim_percent)
    num_layers = len(list(params_dict.values())[0])
    agg_params = []
    for layer_idx in range(num_layers):
        layer_params = np.array([params_dict[cid][layer_idx].flatten() for cid in params_dict])
        if num_trim >= len(layer_params) // 2:
            agg_params.append(np.median(layer_params, axis=0))
        else:
            sorted_params = np.sort(layer_params, axis=0)
            trimmed_params = sorted_params[num_trim:-num_trim] if num_trim > 0 else sorted_params
            agg_params.append(np.mean(trimmed_params, axis=0).reshape(list(params_dict.values())[0][layer_idx].shape))
    return agg_params


#60-70%
#80%
# aggregated = geometric_median_aggregation(filtered_params)
def geometric_median_aggregation(params_dict: Dict[str, List[np.ndarray]], eps=1e-5, max_iter=200) -> Optional[List[np.ndarray]]:
    if not params_dict:
        print("[ERROR] params_dict is empty.")
        return None

    num_layers_list = [len(params) for params in params_dict.values()]
    if len(set(num_layers_list)) != 1:
        print(f"[ERROR] Number of layers inconsistent across clients: {num_layers_list}")
        return None

    num_layers = num_layers_list[0]
    for layer_idx in range(num_layers):
        shapes = [params_dict[cid][layer_idx].shape for cid in params_dict]
        if len(set(shapes)) != 1:
            print(f"[ERROR] Layer {layer_idx}: Parameter shapes are inconsistent: {shapes}")
            return None

    agg_params = []
    for layer_idx in range(num_layers):
        layer_params_list = [params_dict[cid][layer_idx].flatten() for cid in params_dict]
        param_lengths = [len(p) for p in layer_params_list]
        if len(set(param_lengths)) != 1:
            print(f"[ERROR] Layer {layer_idx}: Parameter lengths are inconsistent: {param_lengths}")
            return None

        layer_params = np.array(layer_params_list)
        # 使用加權平均作為初始值
        initial_y = np.average(layer_params, axis=0, weights=np.ones(len(params_dict)) / len(params_dict))

        y = initial_y
        i = 0
        while i < max_iter:
            y_2d = y.reshape(1, -1)
            if y_2d.shape != (1, param_lengths[0]):
                print(f"[ERROR] Layer {layer_idx}, Iteration {i}: y_2d shape {y_2d.shape} does not match expected (1, {param_lengths[0]})")
                return None

            D = cdist(layer_params, y_2d, metric='euclidean')
            weights = 1.0 / (D + eps)
            weights = weights / np.sum(weights)

            y_new = np.sum(weights * layer_params, axis=0)
            if y_new.shape != (param_lengths[0],):
                print(f"[ERROR] Layer {layer_idx}, Iteration {i}: y_new shape {y_new.shape} does not match expected ({param_lengths[0]},)")
                return None

            if np.all(np.abs(y_new - y) < eps):
                break
            y = y_new
            i += 1

        if i == max_iter - 1:
            print(f"[WARNING] Layer {layer_idx}: Reached max iterations ({max_iter}) without convergence")
        
        agg_params.append(y.reshape(list(params_dict.values())[0][layer_idx].shape))

    return agg_params

#爛
# aggregated = rfa_aggregation(filtered_params, clip_norm=dynamic_clip_norm)
def rfa_aggregation(params_dict: Dict[str, List[np.ndarray]], clip_norm: float = 5.0) -> Optional[List[np.ndarray]]:
    """
    Robust Federated Averaging，基於剪裁和加權平均。
    """
    if not params_dict:
        return None
    num_layers = len(list(params_dict.values())[0])
    agg_params = []
    for layer_idx in range(num_layers):
        layer_params = np.array([params_dict[cid][layer_idx].flatten() for cid in params_dict])
        norms = np.linalg.norm(layer_params, axis=1)
        max_norm = np.max(norms)
        if max_norm > clip_norm:
            scale = clip_norm / (max_norm + 1e-9)
            layer_params = layer_params * scale
        weights = np.ones(len(params_dict)) / len(params_dict)  # 均等權重，可根據聲譽調整
        agg_params.append(np.average(layer_params, weights=weights, axis=0).reshape(list(params_dict.values())[0][layer_idx].shape))
    return agg_params


def sanitize_aggregation(
    client_params_list: Dict[str, List[np.ndarray]],
    client_info_table: Dict[str, Dict[str, float]],
    w_st: float = 0.2,
    w_lt: float = 0.2,
    w_sim: float = 0.3,
    w_shap: float = 0.3,
    temp: float = 0.3,
    min_weight: float = 1e-2,
    noise_sigma: float = 0.1,
    clip_norm: float = 5.0,
) -> Optional[List[np.ndarray]]:
    if not client_params_list:
        return None
    
    cids = list(client_params_list.keys())
    
    update_norms = {}
    for cid, params in client_params_list.items():
        if "previous_params" in client_info_table[cid] and client_info_table[cid]["previous_params"] is not None:
            prev_params = client_info_table[cid]["previous_params"]
            norm = _l2_distance(params, prev_params)
            update_norms[cid] = norm
        else:
            update_norms[cid] = 0.0
    
    norm_values = list(update_norms.values())
    q1, q3 = np.percentile(norm_values, [25, 75])
    iqr = q3 - q1
    dynamic_clip_norm = max(1e-6, min(clip_norm, q3 + 1.5 * iqr))
    
    clipped_params_list = {}
    for cid, params in client_params_list.items():
        norm = update_norms[cid]
        if norm > dynamic_clip_norm:
            scale = dynamic_clip_norm / (norm + 1e-9)
            clipped_params = [p * scale for p in params]
            clipped_params_list[cid] = clipped_params
            print(f"[INFO] 客戶端 {cid} 更新被剪裁：範數 {norm:.4f} > {dynamic_clip_norm:.4f}")
        else:
            clipped_params_list[cid] = params
    
    suspicious_count = sum(1 for cid in cids if client_info_table[cid].get("is_suspicious", False))
    suspicious_ratio = suspicious_count / len(cids) if cids else 0.0
    
    if suspicious_ratio > 0.3:
        w_sim_adjusted = w_sim + 0.5
        w_shap_adjusted = w_shap + 0.3
        w_st_adjusted = w_st - 0.3
        w_lt_adjusted = w_lt - 0.3
    else:
        w_sim_adjusted = w_sim - 0.05
        w_shap_adjusted = w_shap
        w_st_adjusted = w_st
        w_lt_adjusted = w_lt + 0.15
    total_w = w_st_adjusted + w_lt_adjusted + w_sim_adjusted + w_shap_adjusted
    if total_w > 0:
        w_st_adjusted /= total_w
        w_lt_adjusted /= total_w
        w_sim_adjusted /= total_w
        w_shap_adjusted /= total_w
    
    st = np.array([client_info_table[c].get("short_term_rs", 0) for c in cids])
    lt = np.array([client_info_table[c].get("long_term_rs", 0) for c in cids])
    sim = np.array([max(0, client_info_table[c].get("similarity", 0)) for c in cids])
    shp = np.array([max(0, client_info_table[c].get("shapley", 0)) for c in cids])
    norms = np.array([update_norms[c] for c in cids])
    
    def _normalize(arr):
        arr = np.clip(arr, 0, None)
        return np.ones_like(arr) if arr.max() - arr.min() < 1e-6 else (arr - arr.min()) / (arr.max() - arr.min())
    
    st_n, lt_n, sim_n, shp_n = map(_normalize, (st, lt, sim, shp))
    norm_penalty = np.exp(-norms / dynamic_clip_norm)
    suspicion_penalty = np.array([0.2 if client_info_table[cid].get("is_suspicious", False) else 1.0 for cid in cids])
    
    sim_variance = np.var(sim)
    non_iid_penalty = np.exp(-sim_variance / 0.1) if sim_variance > 0 else 1.0
    score = (w_st_adjusted * st_n + w_lt_adjusted * lt_n + w_sim_adjusted * sim_n + w_shap_adjusted * shp_n) * norm_penalty * suspicion_penalty * non_iid_penalty
    exp_score = np.exp(score / max(temp, 1e-6))
    weights = exp_score / exp_score.sum()
    
    if np.any(np.isnan(weights)):
        print("[WARNING] NaN weights detected, setting to uniform weights")
        weights = np.ones_like(weights) / len(weights)
    
    smooth_factor = 0.5 if suspicious_ratio < 0.3 else 0.3
    for i, cid in enumerate(cids):
        prev_weight = client_info_table[cid].get("last_weight", weights[i])
        weights[i] = (1 - smooth_factor) * weights[i] + smooth_factor * prev_weight
        client_info_table[cid]["last_weight"] = weights[i]
    
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma * (1 + suspicious_ratio), size=weights.shape)
        weights += noise
        weights = np.clip(weights, 0, None)
        if weights.sum() == 0:
            weights = np.ones_like(weights) / len(weights)
        weights /= weights.sum()
    
    for idx, cid in enumerate(cids):
        if weights[idx] < min_weight:
            weights[idx] = min_weight
        pid = client_info_table[cid].get("partition_id", cid)
        print(f"[INFO] 客戶端 {pid} 權重：{weights[idx]:.4f}")
    
    if weights.sum() == 0:
        print("[WARNING] All weights are zero, setting to uniform weights")
        weights = np.ones_like(weights) / len(weights)
    
    weights /= weights.sum()
    
    filtered_params = {cid: clipped_params_list[cid] for idx, cid in enumerate(cids) if weights[idx] >= min_weight}
    if not filtered_params:
        print("[WARNING] No filtered params, using all params with uniform weights")
        filtered_params = clipped_params_list
        weights = np.ones(len(cids)) / len(cids)
    
    num_layers = len(list(clipped_params_list.values())[0])
    aggregated = None
    
    # 動態選擇聚合方法
    if suspicious_ratio > 0.3:
        print(f"[INFO] High suspicious ratio ({suspicious_ratio:.2f}), using Bulyan aggregation.")
        aggregated = bulyan_aggregation(filtered_params, num_reject=int(suspicious_ratio * len(cids)))
    else:
        print(f"[INFO] Normal suspicious ratio ({suspicious_ratio:.2f}), using Geometric Median aggregation.")
        aggregated = geometric_median_aggregation(filtered_params)
    
    if aggregated is None:
        print("[WARNING] Aggregation failed, falling back to simple averaging")
        aggregated = []
        for layer_idx in range(num_layers):
            layer_params = np.array([clipped_params_list[cid][layer_idx].flatten() for cid in clipped_params_list])
            agg_layer = np.average(layer_params, axis=0, weights=weights)
            aggregated.append(agg_layer.reshape(clipped_params_list[cids[0]][layer_idx].shape))
    
    print(f"[INFO] 聚合完成，使用 {len(filtered_params)} 個客戶端的參數。")
    return aggregated

def multi_krum_aggregation(params_dict: Dict[str, List[np.ndarray]], n_selected: int = None, device: str = "cuda", block_size: int = 1000000) -> List[List[np.ndarray]]:
    """
    Multi-Krum 聚合方法：預設假設惡意客戶端數 f = floor(n/2) - 1，使用分塊計算距離。
    
    Args:
        params_dict: 客戶端參數字典，鍵為客戶端 ID，值為參數列表（numpy 數組）
        n_selected: 要選擇的客戶端更新數量（若為 None，則設為 n - 2f）
        device: 計算設備（"cuda" 或 "cpu"）
        block_size: 每塊參數的維度大小（默認 1M）
        
    Returns:
        選中的客戶端參數列表（List[List[np.ndarray]]）
    """
    # 檢查參數是否有效
    for cid, params in params_dict.items():
        for param in params:
            if np.any(np.isnan(param)) or np.any(np.isinf(param)):
                print(f"警告: 客戶端 {cid} 的參數包含無效值 (NaN 或 Inf)")
                return [params_dict[list(params_dict.keys())[0]]]
    
    # 獲取客戶端數量
    cids = list(params_dict.keys())
    n = len(cids)
    
    # 預設 f = floor(n/2) - 1
    f = (n // 2) - 1
    if f < 0:
        f = 0  # 確保 f >= 0（如 n=2 時）
    
    # 設置 n_selected（若未提供）
    if n_selected is None:
        n_selected = n - 2 * f
    if n_selected <= 0:
        print(f"警告: 選擇的客戶端數量 ({n_selected}) 無效，調整 f")
        f = (n - 1) // 2
        n_selected = n - 2 * f
    if n_selected > n:
        raise ValueError(f"選擇的客戶端數量 ({n_selected}) 不能大於總客戶端數量 ({n})")
    
    # 正規化客戶端參數
    normalized_params_dict = {}
    for cid, params in params_dict.items():
        norm = np.sqrt(sum(np.sum(param ** 2) for param in params))
        if norm > 1e-8:  # 避免除零
            normalized_params_dict[cid] = [param / norm for param in params]
        else:
            normalized_params_dict[cid] = params
            print(f"警告: 客戶端 {cid} 參數範數過小 ({norm})")
    
    # 將參數展平並轉為 PyTorch 張量
    flattened_params = [
        torch.tensor(np.concatenate([param.flatten() for param in normalized_params_dict[cid]]), device=device, dtype=torch.float32)
        for cid in cids
    ]
    flattened_params = torch.stack(flattened_params)  # 形狀：(n, d)
    
    # 初始化距離矩陣
    distances = torch.zeros((n, n), device=device)
    d = flattened_params.shape[1]
    
    # 分塊計算距離
    for i in range(0, d, block_size):
        block = flattened_params[:, i:i+block_size]  # 形狀：(n, block_size)
        diff = block.unsqueeze(1) - block.unsqueeze(0)  # 形狀：(n, n, block_size)
        distances += torch.sum(diff ** 2, dim=-1)  # 累加平方和
        torch.cuda.empty_cache()  # 釋放臨時記憶體
    
    distances = torch.sqrt(distances)  # 形狀：(n, n)
    distances.fill_diagonal_(float('inf'))
    
    # 計算 Krum 分數
    sorted_distances, _ = torch.sort(distances, dim=1)
    scores = torch.sum(sorted_distances[:, :n - n_selected + 1], dim=1)
    
    # 選擇得分最低的 n_selected 個客戶端
    _, selected_indices = torch.topk(scores, n_selected, largest=False)
    selected_cids = [cids[i.item()] for i in selected_indices]
    
    # 返回選中的原始參數（非正規化）
    selected_params = [params_dict[cid] for cid in selected_cids]
    
    return selected_params

def bulyan_aggregation(params_dict: Dict[str, List[np.ndarray]], f: int = None, beta: float = 0.2, device: str = "cuda") -> List[np.ndarray]:
    """
    Bulyan 聚合方法：使用 PyTorch 和 GPU 加速。
    
    Args:
        params_dict: 客戶端參數字典，鍵為客戶端 ID，值為參數列表（numpy 數組）
        f: 假設的惡意客戶端數量，若為 None，則設為 n // 3
        beta: Trimmed Mean 中裁剪的比例
        device: 計算設備（"cuda" 或 "cpu"）
        
    Returns:
        聚合後的參數（numpy 數組列表）
    """
    n = len(params_dict)
    if f is None:
        f = n // 3
    if f >= n / 2:
        raise ValueError(f"惡意客戶端數量 f ({f}) 必須小於客戶端總數的一半 ({n / 2})")
    
    n_selected = n - 2 * f
    if n_selected <= 0:
        raise ValueError(f"選擇的客戶端數量 ({n_selected}) 必須大於 0")
    
    # 1) 使用 Multi-Krum 選擇 n - 2f 個客戶端
    selected_params = multi_krum_aggregation(params_dict, n_selected=n_selected, device=device)
    selected_params_dict = {f"selected_{i}": params for i, params in enumerate(selected_params)}
    
    # 2) Trimmed Mean 聚合
    num_layers = len(list(selected_params_dict.values())[0])
    agg_params = []
    
    for layer_idx in range(num_layers):
        # 轉為 PyTorch 張量
        layer_params = [
            torch.tensor(selected_params_dict[cid][layer_idx].flatten(), device=device, dtype=torch.float32)
            for cid in selected_params_dict
        ]
        layer_params = torch.stack(layer_params)  # 形狀：(m, d_layer)
        
        # 動態調整裁剪數量
        num_params = layer_params.shape[0]
        num_to_remove = min(int(beta * num_params), (num_params - 1) // 2)
        if num_to_remove * 2 >= num_params:
            print(f"警告: 層 {layer_idx} 的裁剪數量 ({num_to_remove}) 過多，調整為 0")
            num_to_remove = 0
        
        # 對每個維度進行排序和裁剪
        if num_to_remove == 0:
            trimmed_mean = torch.mean(layer_params, dim=0)
        else:
            sorted_params, _ = torch.sort(layer_params, dim=0)
            trimmed_params = sorted_params[num_to_remove:-num_to_remove]
            trimmed_mean = torch.mean(trimmed_params, dim=0)
        
        # 轉回 NumPy 並重塑
        agg_params.append(trimmed_mean.cpu().numpy().reshape(list(selected_params_dict.values())[0][layer_idx].shape))
    
    return agg_params



