import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler  # 使用 torch.cuda.amp 模組


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train(net, trainloader, epochs: int, verbose=False):
    """正常訓練函式：使用 CrossEntropyLoss + label smoothing"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs * len(trainloader)))
    scaler = GradScaler("cuda", enabled=(DEVICE == "cuda"))
    
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
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
            
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
        
        scheduler.step()
        train_loss = epoch_loss / len(trainloader.dataset)
        train_accuracy = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {train_loss:.4f}, accuracy {train_accuracy:.4f}")
    return train_loss, train_accuracy

def UPA_train(net, trainloader, epochs: int, alpha: float = 100.0, attack_steps: int = 9, verbose=False):
    """
    UPA_train：正常訓練後再對一個批次資料做多步驟梯度上升，
    使模型參數朝著增加損失的方向更新以破壞模型效能。
    注意：攻擊階段禁用 weight_decay，並進行多步攻擊更新。
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 正常訓練用 optimizer（可保留 weight_decay）
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs * len(trainloader)))
    scaler = GradScaler("cuda", enabled=(DEVICE == "cuda"))
    
    net.train()
    # 正常訓練階段
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
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
            
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(trainloader.dataset)
        acc = correct / total
        if verbose:
            print(f"[Normal] Epoch {epoch+1}: train loss {avg_loss:.4f}, accuracy {acc:.4f}")
    
    # -----------------------------
    # 攻擊階段：多步驟梯度上升
    # -----------------------------
    net.train()
    try:
        batch = next(iter(trainloader))
    except StopIteration:
        return  # 若無資料則跳過攻擊

    images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
    images = images.to(DEVICE, memory_format=torch.channels_last)
    
    # 攻擊階段使用較高的攻擊強度，並禁用 weight_decay（設為 0）
    attack_optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0)
    # 如果需要，可額外固定攻擊階段的學習率
    attack_scaler = GradScaler("cuda", enabled=(DEVICE == "cuda"))
    
    for step in range(attack_steps):
        attack_optimizer.zero_grad()
        with autocast("cuda"):
            outputs = net(images)
            loss = criterion(outputs, labels)
        neg_loss = -alpha * loss  # 梯度上升
        attack_scaler.scale(neg_loss).backward()
        attack_scaler.step(attack_optimizer)
        attack_scaler.update()
        if verbose:
            print(f"[UPA Attack] Step {step+1}/{attack_steps}, normal loss: {loss.item():.4f}")

def TPA_train(net, trainloader, epochs: int, original_label: int = 0, target_label: int = 2,
              poison_num: int = 20, alpha: float = 1.0, attack_steps: int = 9, verbose=False):
    """
    TPA_train：正常訓練 epochs 輪後，
    從 trainloader 中找出指定 original_label 的樣本（最多 poison_num 筆），
    將其標籤修改為 target_label，並用多步驟梯度下降（乘上 alpha 放大攻擊效果）
    更新模型參數，達到 targeted poisoning 的效果。
    
    注意：攻擊階段禁用 weight_decay，同時建議增加攻擊步驟使效果累積。
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs * len(trainloader)))
    scaler = GradScaler("cuda", enabled=(DEVICE == "cuda"))
    
    net.train()
    # 正常訓練階段
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
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
            
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(trainloader.dataset)
        acc = correct / total
        if verbose:
            print(f"[Train] Epoch {epoch+1}/{epochs}: loss {avg_loss:.4f}, accuracy {acc:.4f}")
    
    # -----------------------------
    # 攻擊階段：找出 poison 樣本並進行多步驟 targeted poisoning 更新
    # -----------------------------
    net.train()
    poison_imgs = []
    poison_labels = []
    count = 0

    # 重新遍歷 trainloader 找出屬於 original_label 的樣本
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
        if verbose:
            print(f"[TPA] No samples found for original_label {original_label}. Skipping attack.")
        return

    poison_imgs = torch.stack(poison_imgs, dim=0)   # shape: [N, C, H, W]
    poison_labels = torch.stack(poison_labels, dim=0)  # shape: [N]
    fake_labels = torch.full_like(poison_labels, target_label).to(DEVICE)

    # 攻擊階段使用獨立的 optimizer（禁用 weight_decay）
    attack_optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0)
    attack_scaler = GradScaler("cuda", enabled=(DEVICE == "cuda"))

    for step in range(attack_steps):
        attack_optimizer.zero_grad()
        with autocast("cuda"):
            outputs = net(poison_imgs)
            loss = criterion(outputs, fake_labels) * alpha  # 放大攻擊效果
        attack_scaler.scale(loss).backward()
        attack_scaler.step(attack_optimizer)
        attack_scaler.update()
        if verbose:
            print(f"[TPA Attack] Step {step+1}/{attack_steps}, poison loss: {loss.item():.4f}")
