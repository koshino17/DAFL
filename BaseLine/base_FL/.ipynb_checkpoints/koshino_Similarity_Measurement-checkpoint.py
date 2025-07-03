from typing import List
import numpy as np

import torch
import torch.nn.functional as F

def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    計算向量 vec1 與 vec2 的 cos 相似度
    """
    # 若張量的維度不只一階，可先攤平成一維
    v1 = vec1.view(-1)
    v2 = vec2.view(-1)

    # 避免 0 向量分母報錯
    if v1.norm(p=2) == 0 or v2.norm(p=2) == 0:
        return 0.0

    # F.cosine_similarity 返回 tensor，取 item() 得到純量
    sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    return sim

def pearson_correlation(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    計算向量 vec1 與 vec2 的 Pearson correlation
    """
    v1 = vec1.view(-1)
    v2 = vec2.view(-1)

    if v1.numel() < 2:
        return 0.0

    # 去均值
    v1_centered = v1 - torch.mean(v1)
    v2_centered = v2 - torch.mean(v2)

    denominator = (v1_centered.norm(p=2) * v2_centered.norm(p=2)).item()
    if denominator == 0:
        return 0.0

    corr = torch.sum(v1_centered * v2_centered).item() / denominator
    return corr

# ------------------ 壓縮與相似度檢測 ------------------
def compress_parameters(numpy_params: List[np.ndarray], compress_length: int = 100) -> np.ndarray:
    """
    將模型參數壓縮成一個一維向量，
    對每個參數攤平後取前 compress_length 個元素，再串接起來。
    """
    compressed_list = []
    for p in numpy_params:
        flat = p.flatten()
        k = min(compress_length, flat.shape[0])
        compressed_list.append(flat[:k])
    return np.concatenate(compressed_list, axis=0)

def _compute_similarity_compressed(global_params: List[np.ndarray],
                                   local_params: List[np.ndarray],
                                   compress_length: int = 100) -> float:
    """
    將 global_params 與 local_params 分別展平成一個向量，
    然後只取前 compress_length 個元素計算 cosine similarity 與 Pearson correlation 的平均，
    作為最終的相似度分數。
    """
    # 將所有參數展平並串接成一個長向量
    global_flat = np.concatenate([p.flatten() for p in global_params])
    local_flat = np.concatenate([p.flatten() for p in local_params])
    k = min(compress_length, len(global_flat), len(local_flat))
    global_sub = global_flat[:k]
    local_sub = local_flat[:k]
    global_tensor = torch.from_numpy(global_sub).float()
    local_tensor = torch.from_numpy(local_sub).float()
    cos_sim = torch.nn.functional.cosine_similarity(global_tensor.unsqueeze(0),
                                                     local_tensor.unsqueeze(0)).item()
    pearson_sim = pearson_correlation(global_tensor, local_tensor)
    return 0.5 * (cos_sim + pearson_sim)