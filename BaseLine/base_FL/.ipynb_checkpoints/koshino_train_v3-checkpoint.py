import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler  # 使用 torch.cuda.amp 模組
from koshino_FL.koshino_Lookahead import Lookahead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def train(net, trainloader, epochs: int, verbose=False):
    """正常訓練函式：使用 CrossEntropyLoss + label smoothing"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    base_optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer = Lookahead(base_optimizer, alpha=0.5, k=5)
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
        
        optimizer.step()
        scheduler.step()
        train_loss = epoch_loss / len(trainloader.dataset)
        train_accuracy = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {train_loss:.4f}, accuracy {train_accuracy:.4f}")
    return train_loss, train_accuracy

def UPA_train(net, trainloader, epochs: int, alpha: float = 100.0, attack_steps: int = 9, verbose=False):
    """
    UPA_train：
    先使用一般的 train() 進行正常訓練，再對一個 batch 資料進行多步驟梯度上升攻擊，
    使模型參數朝著增加損失的方向更新，以破壞模型效能。
    最後以 test() 計算並回傳訓練階段的最終 loss 與 accuracy。
    """
    # 正常訓練階段（直接呼叫一般的 train 函式）
    train_loss, train_accuracy = train(net, trainloader, epochs, verbose)
    
    # -----------------------------
    # 攻擊階段：對單一 batch 進行多步驟梯度上升
    # -----------------------------
    net.train()
    try:
        batch = next(iter(trainloader))
    except StopIteration:
        return train_loss, train_accuracy  # 若無資料則跳過攻擊

    images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
    images = images.to(DEVICE, memory_format=torch.channels_last)
    
    # 攻擊階段使用獨立的 optimizer（禁用 weight_decay）
    attack_optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0)
    attack_scaler = GradScaler("cuda", enabled=(DEVICE == "cuda"))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
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
    
    # 攻擊完成後，重新以 test() 計算模型在訓練集上的 loss 與 accuracy
    final_loss, final_accuracy = test(net, trainloader)
    return final_loss, final_accuracy

def TPA_train(net, trainloader, epochs: int, original_label: int = 0, target_label: int = 2,
              poison_num: int = 20, alpha: float = 1.0, attack_steps: int = 9, verbose=False):
    """
    TPA_train：
    先使用一般的 train() 進行正常訓練，再從 trainloader 中找出指定 original_label 的樣本（最多 poison_num 筆），
    將其標籤改為 target_label，並用多步驟梯度下降（乘上 alpha）進行 targeted poisoning 攻擊，
    最後以 test() 計算模型在訓練集上的最終 loss 與 accuracy。
    """
    # 正常訓練階段
    train_loss, train_accuracy = train(net, trainloader, epochs, verbose)
    
    # -----------------------------
    # 攻擊階段：從 trainloader 中找出 poison 樣本
    # -----------------------------
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
        if verbose:
            print(f"[TPA] No samples found for original_label {original_label}. Skipping attack.")
        return train_loss, train_accuracy

    poison_imgs = torch.stack(poison_imgs, dim=0)   # shape: [N, C, H, W]
    poison_labels = torch.stack(poison_labels, dim=0)  # shape: [N]
    fake_labels = torch.full_like(poison_labels, target_label).to(DEVICE)

    # 攻擊階段使用獨立的 optimizer（禁用 weight_decay）
    attack_optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0)
    attack_scaler = GradScaler("cuda", enabled=(DEVICE == "cuda"))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

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

    # 攻擊完成後，重新以 test() 計算模型在訓練集上的 loss 與 accuracy
    final_loss, final_accuracy = test(net, trainloader)
    return final_loss, final_accuracy
