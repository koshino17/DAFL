import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler  # 使用 torch.cuda.amp 模組
from koshino_FL.koshino_Lookahead import Lookahead
import copy
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # ------------------ 測試函式 ------------------
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
# ------------------ Ditto 訓練 ------------------
def train(net, trainloader, config, lambda_reg=0.05, mu=0.5, verbose=False):
    """
    結合 Ditto 和 Lookahead 的訓練方法，適應 Non-IID 數據。
    使用 Ditto 的個人化正則化與 Lookahead 優化器，帶學習率調度。
    回傳 (final_loss, global_params)。
    """
    epochs = config["local_epochs"]
    initial_lr = 0.01  # 可根據需求調整
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    base_optimizer = torch.optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)
    optimizer = Lookahead(base_optimizer, alpha=0.5, k=5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)  # 輕微衰減
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    # 初始化全局模型參數 (w)
    global_params = [val.detach().clone() for val in net.parameters()]
    net.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in trainloader:
            images = batch["img"].to(DEVICE, memory_format=torch.channels_last)
            labels = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = net(images)
                loss = criterion(outputs, labels)
                # Ditto 正則化項
                prox_ditto = 0.0
                for w, gw in zip(net.parameters(), global_params):
                    prox_ditto += (lambda_reg / 2) * ((w - gw.to(w.device)).norm() ** 2)
                # Lookahead 近端正則項
                prox_lookahead = 0.0
                if global_params is not None and mu > 0:
                    for w, gw in zip(net.parameters(), global_params):
                        prox_lookahead += (mu / 2.0) * ((w - gw.to(w.device)).norm() ** 2)
                loss = loss + prox_ditto + prox_lookahead
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        scheduler.step()
        if verbose:
            avg_loss = running_loss / len(trainloader)
            print(f"[Ditto+Lookahead Train] Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    final_loss, train_acc = test(net, trainloader)
    if verbose:
        print(f"[Ditto+Lookahead Train] Final Loss: {final_loss:.4f}, Train Acc: {train_acc:.4f}")
    return final_loss, global_params

# ------------------ 測試函式 ------------------
# def test(net, testloader):
#     """Evaluate the network on the entire test set."""
#     criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
#     correct, total, loss = 0, 0, 0.0
#     net.eval()

#     with torch.no_grad():
#         for batch in testloader:
#             images = batch["img"].to(DEVICE, memory_format=torch.channels_last)
#             labels = batch["label"].to(DEVICE)

#             # 確保 labels 是 1D 張量
#             if labels.dim() > 1:
#                 labels = labels.squeeze()
#             if labels.dim() != 1:
#                 raise ValueError(f"Expected labels to be 1D tensor, but got shape {labels.shape}")

#             with autocast("cuda"):
#                 outputs = net(images)  # 移除 shared_only 參數
#                 loss += criterion(outputs, labels).item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     loss /= len(testloader.dataset)
#     accuracy = correct / total
#     return loss, accuracy
# # ------------------ Ditto 訓練 ------------------
# def train(net, trainloader, config, lambda_reg=0.05, verbose=False):
#     """
#     結合 Ditto 的訓練方法，適應 Non-IID 數據，支援 FedPer 分層結構。
#     模型分為共享層和個人化層，僅對共享層應用 Ditto 正則化。
#     回傳 (final_loss, shared_params)。
#     """
#     epochs = config["local_epochs"]
#     initial_lr = 0.01
#     criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
#     base_optimizer = torch.optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)
#     optimizer = Lookahead(base_optimizer, alpha=0.5, k=5)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
#     scaler = GradScaler(enabled=(DEVICE == "cuda"))

#     # 提取共享層參數
#     shared_params = [val.detach().clone() for name, val in net.named_parameters() if "shared" in name]
#     net.train()

#     for epoch in range(epochs):
#         running_loss = 0.0
#         for batch in trainloader:
#             images = batch["img"].to(DEVICE, memory_format=torch.channels_last)
#             labels = batch["label"].to(DEVICE)
#             optimizer.zero_grad()
#             with autocast("cuda"):
#                 outputs = net(images)
#                 loss = criterion(outputs, labels)
#                 # Ditto 正則化項（僅應用於共享層）
#                 prox_ditto = 0.0
#                 for (name, w), gw in zip(
#                     [(n, p) for n, p in net.named_parameters() if "shared" in n],
#                     shared_params
#                 ):
#                     prox_ditto += (lambda_reg / 2) * ((w - gw.to(w.device)).norm() ** 2)
#                 loss = loss + prox_ditto
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             running_loss += loss.item()
#         scheduler.step()
#         if verbose:
#             avg_loss = running_loss / len(trainloader)
#             print(f"[Ditto Train] Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

#     final_loss, train_acc = test(net, trainloader)
#     if verbose:
#         print(f"[Ditto Train] Final Loss: {final_loss:.4f}, Train Acc: {train_acc:.4f}")
#     return final_loss, shared_params

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


