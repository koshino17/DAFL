from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler  # 使用 torch.cuda.amp 模組
from koshino_FL.koshino_Similarity_Measurement import cosine_similarity, pearson_correlation
from koshino_FL.koshino_Lookahead import Lookahead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# ------------------ 測試函式 ------------------
def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            images = images.to(DEVICE, memory_format=torch.channels_last)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

# ------------------ Lookahead 正常訓練 ------------------
def lookahead_train(net, parameters, trainloader, epochs: int, verbose=False):
    """使用 Lookahead 進行正常訓練，並回傳最終模型參數與 slow update 相似度。"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    base_optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer = Lookahead(base_optimizer, alpha=0.5, k=5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs * len(trainloader)))
    scaler = GradScaler(enabled=(DEVICE=="cuda"))
    
    net.train()
    for epoch in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            images = images.to(DEVICE, memory_format=torch.channels_last)
            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = net(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} completed.")
    # 取得 Lookahead 的慢速權重
    slow_params = optimizer.get_slow_weights()
    # 計算慢速權重與全局參數間的壓縮相似度
    similarity = _compute_similarity_compressed(parameters, slow_params, compress_length=100)
    final_loss, _ = test(net, trainloader)
    return final_loss, similarity

# ------------------ Lookahead UPA (Un-targeted Poisoning Attack) ------------------
def lookahead_UPA_train(net, parameters, trainloader, epochs: int, attack_steps: int = 9, attack_alpha: float = 100.0, verbose=False):
    """
    使用 Lookahead 進行正常訓練，再對一個 batch 執行 UPA 攻擊（梯度上升），
    回傳最終模型參數與慢更新相似度。
    """
    # 正常訓練階段 (Lookahead)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    base_optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer = Lookahead(base_optimizer, alpha=0.5, k=5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs * len(trainloader)))
    scaler = GradScaler(enabled=(DEVICE=="cuda"))
    
    net.train()
    for epoch in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            images = images.to(DEVICE, memory_format=torch.channels_last)
            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = net(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} completed.")
    
    # 正常訓練完成後，從 Lookahead 取慢速權重
    slow_params = optimizer.get_slow_weights()
    similarity = _compute_similarity_compressed(parameters, slow_params, compress_length=100)
    
    # 攻擊階段：對單一 batch 進行梯度上升攻擊
    net.train()
    try:
        batch = next(iter(trainloader))
    except StopIteration:
        final_loss, _ = test(net, trainloader)
        return get_parameters(net), {"loss": float(final_loss), "lookahead_similarity": similarity}
    images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
    images = images.to(DEVICE, memory_format=torch.channels_last)
    
    attack_optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0)
    attack_scaler = GradScaler(enabled=(DEVICE=="cuda"))
    for step in range(attack_steps):
        attack_optimizer.zero_grad()
        with autocast("cuda"):
            outputs = net(images)
            loss = criterion(outputs, labels)
        neg_loss = -attack_alpha * loss
        attack_scaler.scale(neg_loss).backward()
        attack_scaler.step(attack_optimizer)
        attack_scaler.update()
        if verbose:
            print(f"[UPA Attack] Step {step+1}/{attack_steps} completed, loss={loss.item():.4f}")
    
    final_loss, _ = test(net, trainloader)
    return final_loss, similarity

# ------------------ Lookahead TPA (Targeted Poisoning Attack) ------------------
def lookahead_TPA_train(net, parameters, trainloader, epochs: int, original_label: int = 0, target_label: int = 2,
                        poison_num: int = 20, attack_steps: int = 9, attack_alpha: float = 1.0, verbose=False):
    """
    使用 Lookahead 進行正常訓練，再從 trainloader 中挑出指定 original_label 的樣本，
    將其標籤改為 target_label，並進行多步驟梯度下降攻擊 (targeted poisoning)。
    回傳最終模型參數與慢更新相似度。
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    base_optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer = Lookahead(base_optimizer, alpha=0.5, k=5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs * len(trainloader)))
    scaler = GradScaler(enabled=(DEVICE=="cuda"))
    
    net.train()
    for epoch in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            images = images.to(DEVICE, memory_format=torch.channels_last)
            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = net(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} completed.")
    
    slow_params = optimizer.get_slow_weights()
    similarity = _compute_similarity_compressed(parameters, slow_params, compress_length=100)
    
    # 攻擊階段：從 trainloader 找出 poison 樣本 (original_label)
    net.train()
    poison_imgs = []
    poison_labels = []
    count = 0
    for batch in trainloader:
        imgs, lbs = batch["img"].to(DEVICE, memory_format=torch.channels_last), batch["label"].to(DEVICE)
        for i in range(len(imgs)):
            if lbs[i].item() == original_label:
                poison_imgs.append(imgs[i])
                poison_labels.append(lbs[i])
                count += 1
                if count >= poison_num:
                    break
        if count >= poison_num:
            break
    if len(poison_imgs) == 0:
        final_loss, _ = test(net, trainloader)
        return get_parameters(net), {"loss": float(final_loss), "lookahead_similarity": similarity}
    poison_imgs = torch.stack(poison_imgs, dim=0)
    poison_labels = torch.stack(poison_labels, dim=0)
    fake_labels = torch.full_like(poison_labels, target_label).to(DEVICE)
    
    attack_optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0)
    attack_scaler = GradScaler(enabled=(DEVICE=="cuda"))
    for step in range(attack_steps):
        attack_optimizer.zero_grad()
        with autocast("cuda"):
            outputs = net(poison_imgs)
            loss = criterion(outputs, fake_labels) * attack_alpha
        attack_scaler.scale(loss).backward()
        attack_scaler.step(attack_optimizer)
        attack_scaler.update()
        if verbose:
            print(f"[TPA Attack] Step {step+1}/{attack_steps} completed, poison loss={loss.item():.4f}")
    
    final_loss, _ = test(net, trainloader)
    return final_loss, similarity

