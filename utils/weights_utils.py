from collections import OrderedDict
import torch

# def weights_substraction(weights0, weights1):
#     # substraction for for state_dicts
#     ret = OrderedDict()
#     for k in weights0:
#         ret[k] = weights0[k] - weights1[k]
#     return ret


# def norm_l2(weights):
#     ret = 0
#     for k, v in weights.items():
#         value = v.data.norm(2)
#         ret += value.item() ** 2
#     return ret ** 0.5

# def norm(weights):
#     return norm_l2(weights)

def norm_linf(weights):
    ret = 0
    for k, v in weights.items():
        ret = max(ret, v.data.abs().max().item())
    return ret



def weights_substraction(weights1, weights2):
    """安全地計算兩組權重之間的差異"""
    # 靜默轉換，不再打印每次警告
    w1 = OrderedDict(weights1) if isinstance(weights1, dict) and not isinstance(weights1, OrderedDict) else weights1
    w2 = OrderedDict(weights2) if isinstance(weights2, dict) and not isinstance(weights2, OrderedDict) else weights2
    
    # 檢查轉換結果
    if not isinstance(w1, OrderedDict) or not isinstance(w2, OrderedDict):
        print(f"錯誤: weights_substraction 無法處理的參數類型: {type(weights1)} 和 {type(weights2)}")
        return OrderedDict()  # 返回空字典
    
    result = OrderedDict()
    common_keys = set(w1.keys()).intersection(set(w2.keys()))
    
    if not common_keys:
        print("警告: weights_substraction 未發現共同鍵，無法計算差異")
        return OrderedDict()
    
    for key in common_keys:
        try:
            w1_val = w1[key]
            w2_val = w2[key]
            if isinstance(w1_val, torch.Tensor) and isinstance(w2_val, torch.Tensor):
                # 確保兩者都是浮點張量
                w1_float = w1_val.float() if w1_val.dtype != torch.float else w1_val
                w2_float = w2_val.float() if w2_val.dtype != torch.float else w2_val
                if w1_float.shape == w2_float.shape:
                    result[key] = w1_float - w2_float
                else:
                    # 僅在第一次發現形狀不匹配時輸出警告
                    if key not in result:
                        print(f"警告: 鍵 {key} 的張量形狀不匹配: {w1_float.shape} vs {w2_float.shape}")
            else:
                # 如果不是張量，轉換為浮點張量
                w1_tensor = torch.tensor(w1_val, dtype=torch.float)
                w2_tensor = torch.tensor(w2_val, dtype=torch.float)
                if w1_tensor.shape == w2_tensor.shape:
                    result[key] = w1_tensor - w2_tensor
                else:
                    # 僅在第一次發現形狀不匹配時輸出警告
                    if key not in result:
                        print(f"警告: 鍵 {key} 的張量形狀不匹配: {w1_tensor.shape} vs {w2_tensor.shape}")
        except Exception as e:
            print(f"計算鍵 {key} 的差異時出錯: {e}")
    
    return result

def norm(weights):
    """計算權重的 L2 範數，確保所有張量是浮點型的"""
    if isinstance(weights, dict):
        total_norm = 0.0
        for w in weights.values():
            if isinstance(w, torch.Tensor):
                # 轉換為浮點型
                w_float = w.float() if w.dtype != torch.float else w
                total_norm += torch.linalg.vector_norm(w_float).item() ** 2
            else:
                # 如果不是張量，轉換為浮點張量
                w_tensor = torch.tensor(w, dtype=torch.float)
                total_norm += torch.linalg.vector_norm(w_tensor).item() ** 2
        return torch.sqrt(torch.tensor(total_norm)).item()
    else:
        # 非字典類型的處理（列表或其他）
        raise TypeError("norm 需要一個字典作為輸入")

def norm_l2(vec):
    """計算向量的 L2 範數，確保輸入是浮點型的"""
    if isinstance(vec, torch.Tensor):
        # 轉換為浮點型
        vec_float = vec.float() if vec.dtype != torch.float else vec
        return torch.linalg.vector_norm(vec_float).item()
    else:
        # 如果不是張量，轉換為浮點張量
        vec_tensor = torch.tensor(vec, dtype=torch.float)
        return torch.linalg.vector_norm(vec_tensor).item()

