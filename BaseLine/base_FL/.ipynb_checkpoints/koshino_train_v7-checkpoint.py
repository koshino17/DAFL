import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler  # 使用 torch.cuda.amp 模組
from koshino_FL.koshino_Lookahead import Lookahead
import copy
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------ 測試函式 ------------------
def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(DEVICE, memory_format=torch.channels_last)
            labels = batch["label"].to(DEVICE)
            with autocast("cuda"):  # 啟用混合精度
                outputs = net(images)
                loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

# ------------------ Lookahead 正常訓練 ------------------
def lookahead_train(net, trainloader, config, verbose=False):
    """
    使用 Lookahead 進行正常訓練。
    同步全局參數到模型後，使用 Lookahead 優化器更新模型，最後回傳 (final_loss, metrics)
    其中 metrics 包含 slow_similarity 與 final_similarity（此處兩者相同，因無攻擊階段）。
    """
    epochs = config["local_epochs"]
    server_round = config["server_round"]
    initial_lr = 0.1
    decay_factor = 0.95
    current_lr = initial_lr * (decay_factor ** server_round) if server_round is not None else initial_lr
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    base_optimizer = torch.optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
    optimizer = Lookahead(base_optimizer, alpha=0.5, k=5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs * len(trainloader)))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)  # Fixed LR
    scaler = GradScaler(enabled=(DEVICE=="cuda"))
    # local_steps = 0
    global_params = [val.detach().clone() for val in net.parameters()]
    mu = 0.1

    
    net.train()
    for epoch in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(DEVICE, memory_format=torch.channels_last)
            labels = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = net(images)
                loss = criterion(outputs, labels)
                if global_params is not None:
                    prox = 0.0
                    for w, gw in zip(net.parameters(), global_params):
                        prox += ((w - gw.to(w.device)).norm() ** 2)
                    loss = loss + (mu / 2.0) * prox

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # local_steps += 1
        scheduler.step()
        if verbose:
            print(f"[Lookahead Train] Epoch {epoch+1}/{epochs} completed.")
    
    final_loss, train_acc = test(net, trainloader)
    return final_loss

# ------------------ Lookahead UPA 訓練 ------------------
def lookahead_UPA_train(net, trainloader, config, attack_steps: int = 1, attack_alpha: float = 1.0, verbose=False):
    """
    使用 Lookahead 進行正常訓練，再對單一 batch 執行 UPA 攻擊（梯度上升）。
    回傳 final_loss。
    """
    # 執行正常訓練
    final_loss = train(net, trainloader, config, verbose=verbose)

    # 攻擊階段：對單一 batch 執行 UPA 攻擊（不使用 Lookahead）
    net.train()
    try:
        batch = next(iter(trainloader))
    except StopIteration:
        return final_loss

    images = batch["img"].to(DEVICE, memory_format=torch.channels_last)
    labels = batch["label"].to(DEVICE)

    # 使用單獨的攻擊優化器（不更新 slow weights）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    attack_optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0)
    attack_scaler = GradScaler(enabled=(DEVICE == "cuda"))
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

    final_loss, train_acc = test(net, trainloader)
    return final_loss

# ------------------ Lookahead TPA 訓練 ------------------
def lookahead_TPA_train(net, trainloader, config, original_label: int = 0, target_label: int = 2,
                        poison_num: int = 20, attack_steps: int = 1, attack_alpha: float = 1.0, verbose=False):
    """
    使用 Lookahead 進行正常訓練，再從 trainloader 中挑出指定 original_label 的樣本，
    將其標籤改為 target_label，並進行多步驟梯度下降攻擊 (targeted poisoning)。
    回傳 final_loss。
    """
    # 執行正常訓練
    final_loss = train(net, trainloader, config, verbose=verbose)

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
        return final_loss

    poison_imgs = torch.stack(poison_imgs, dim=0)
    poison_labels = torch.stack(poison_labels, dim=0)
    fake_labels = torch.full_like(poison_labels, target_label).to(DEVICE)

    # 攻擊階段
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    attack_optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0)
    attack_scaler = GradScaler(enabled=(DEVICE == "cuda"))
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

    final_loss, train_acc = test(net, trainloader)
    return final_loss

# ------------------ Normal 訓練 for baseline ------------------
# def train(net, trainloader, config, verbose=False):
#     """
#     使用 Lookahead 進行正常訓練。
#     同步全局參數到模型後，使用 Lookahead 優化器更新模型，最後回傳 (final_loss)
#     """
#     epochs = config["local_epochs"]
#     initial_lr = 0.01  # Lowered from 0.1
#     criterion = nn.CrossEntropyLoss(label_smoothing=0.0)  # Match test
#     base_optimizer = torch.optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)
#     optimizer = Lookahead(base_optimizer, alpha=0.5, k=5)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)  # Fixed LR
#     scaler = GradScaler(enabled=(DEVICE=="cuda"))
#     global_params = [val.detach().clone() for val in net.parameters()]
#     mu = 0.3

#     net.train()
#     for epoch in range(epochs):
#         for batch in trainloader:
#             images = batch["img"].to(DEVICE, memory_format=torch.channels_last)
#             labels = batch["label"].to(DEVICE)
#             optimizer.zero_grad()
#             with autocast("cuda"):
#                 outputs = net(images)
#                 loss = criterion(outputs, labels)
#                 if global_params is not None and mu > 0:
#                     prox = 0.0
#                     for w, gw in zip(net.parameters(), global_params):
#                         prox += ((w - gw.to(w.device)).norm() ** 2)
#                     loss = loss + (mu / 2.0) * prox

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         scheduler.step()
#         if verbose:
#             print(f"[Lookahead Train] Epoch {epoch+1}/{epochs} completed.")

#     final_loss, train_acc = test(net, trainloader)
#     if verbose:
#         print(f"[Lookahead Train] Final Loss: {final_loss:.4f}, Train Acc: {train_acc:.4f}")
#     return final_loss

# ------------------ 知識蒸餾訓練 ------------------
# def train_with_distillation(net, teacher_net, trainloader, config, verbose=False):
#     """
#     使用知識蒸餾（Knowledge Distillation）進行訓練。
#     學生模型（net）從教師模型（teacher_net）學習知識。

#     Args:
#         net: 學生模型
#         teacher_net: 教師模型（通常是全局模型）
#         trainloader: 訓練資料加載器
#         config: 包含訓練配置的字典
#         verbose: 是否輸出詳細訓練信息

#     Returns:
#         final_loss: 最終訓練損失
#     """
#     epochs = config.get("local_epochs", 3)
#     server_round = config.get("server_round", 1)
#     temperature = config.get("distill_temperature", 2.0)
#     alpha = config.get("distill_alpha", 0.5)  # 平衡蒸餾損失與真實標籤損失
    
#     # 動態調整 alpha 參數，早期更依賴教師模型
#     if server_round < 5:
#         alpha = min(0.7, alpha + 0.1)  # 提高對教師知識的依賴
#     else:
#         alpha = max(0.3, alpha - 0.05 * (server_round - 5) / 10)  # 隨著訓練進行降低依賴
    
#     # 設定優化器和學習率
#     initial_lr = 0.01
#     decay_factor = 0.95
#     current_lr = initial_lr * (decay_factor ** server_round) if server_round is not None else initial_lr
    
#     criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
#     kl_criterion = nn.KLDivLoss(reduction="batchmean")
    
#     base_optimizer = torch.optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
#     optimizer = Lookahead(base_optimizer, alpha=0.5, k=5)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
#     scaler = GradScaler(enabled=(DEVICE=="cuda"))
    
#     # 設置全局參數用於 proximal term
#     global_params = [val.detach().clone() for val in net.parameters()]
#     mu = config.get("proximal_mu", 0.1)
    
#     # 確保教師模型處於評估模式
#     teacher_net.eval()
#     net.train()
    
#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         for batch in trainloader:
#             images = batch["img"].to(DEVICE, memory_format=torch.channels_last)
#             labels = batch["label"].to(DEVICE)
            
#             optimizer.zero_grad()
            
#             with autocast("cuda"):
#                 # 學生模型前向傳播
#                 student_outputs = net(images)
                
#                 # 硬標籤損失
#                 hard_loss = criterion(student_outputs, labels)
                
#                 # 從教師模型獲取軟標籤（不需要梯度）
#                 with torch.no_grad():
#                     teacher_outputs = teacher_net(images)
                
#                 # 軟標籤損失 (KL散度)
#                 teacher_soft = torch.nn.functional.softmax(teacher_outputs / temperature, dim=1)
#                 student_soft = torch.nn.functional.log_softmax(student_outputs / temperature, dim=1)
#                 soft_loss = kl_criterion(student_soft, teacher_soft) * (temperature ** 2)
                
#                 # 加入 proximal term 約束
#                 prox_term = 0.0
#                 if global_params is not None and mu > 0:
#                     for w, gw in zip(net.parameters(), global_params):
#                         prox_term += ((w - gw.to(w.device)).norm() ** 2)
#                     prox_term = (mu / 2.0) * prox_term
                
#                 # 總損失
#                 total_loss = (1 - alpha) * hard_loss + alpha * soft_loss + prox_term
            
#             # 梯度下降
#             scaler.scale(total_loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
            
#             epoch_loss += total_loss.item()
            
#         scheduler.step()
#         if verbose:
#             avg_loss = epoch_loss / len(trainloader)
#             print(f"[Distillation] Epoch {epoch+1}/{epochs} completed, avg_loss={avg_loss:.4f}")
    
#     # 評估最終損失
#     final_loss, train_acc = test(net, trainloader)
#     if verbose:
#         print(f"[Distillation] Final Loss: {final_loss:.4f}, Train Acc: {train_acc:.4f}, Alpha: {alpha:.2f}")
    
#     return final_loss

# ------------------ Ditto 訓練 ------------------
def train(net, trainloader, config, lambda_reg=0.1, verbose=False):
    epochs = config["local_epochs"]
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = Lookahead(torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9), alpha=0.5, k=5)
    scaler = GradScaler(enabled=(DEVICE=="cuda"))

    # 初始化全局模型參數 (w)
    global_params = [val.detach().clone() for val in net.parameters()]
    net.train()

    for epoch in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = net(images)
                loss = criterion(outputs, labels)
                # Ditto 正則化項
                prox = 0.0
                for w, gw in zip(net.parameters(), global_params):
                    prox += (lambda_reg / 2) * ((w - gw.to(w.device)).norm() ** 2)
                loss = loss + prox
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if verbose:
            print(f"[Ditto Train] Epoch {epoch+1}/{epochs} completed.")

    final_loss, train_acc = test(net, trainloader)
    return final_loss, global_params  # 返回全局參數以供聚合

