from typing import List, Dict
import numpy as np

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

def multi_krum_aggregation(params_dict, num_reject):
            num_clients = len(params_dict)
            if num_clients <= num_reject + 1:
                return list(params_dict.values())[0] if params_dict else None
            
            num_layers = len(list(params_dict.values())[0])
            agg_params = []
            for layer_idx in range(num_layers):
                layer_params = np.array([params_dict[cid][layer_idx].flatten() for cid in params_dict])
                distances = np.zeros((num_clients, num_clients))
                for i in range(num_clients):
                    for j in range(i + 1, num_clients):
                        diff = layer_params[i] - layer_params[j]
                        distances[i, j] = np.sum(diff * diff)
                        distances[j, i] = distances[i, j]
                
                scores = np.sum(distances, axis=1)
                reject_indices = np.argsort(scores)[-num_reject:] if num_reject < num_clients else []
                valid_indices = [i for i in range(num_clients) if i not in reject_indices]
                
                if not valid_indices:
                    agg_params.append(np.zeros_like(layer_params[0]))
                else:
                    valid_params = layer_params[valid_indices]
                    agg_params.append(np.mean(valid_params, axis=0).reshape(list(params_dict.values())[0][layer_idx].shape))
            
            return agg_params