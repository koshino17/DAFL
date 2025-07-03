import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler  # 使用 torch.cuda.amp 模組

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ 測試函式 ------------------
def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    correct, total, loss = 0, 0, 0.0
    net = net.to(DEVICE)
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
def train(net, trainloader, config, verbose=False):
    """
    使用 Lookahead 進行正常訓練。
    同步全局參數到模型後，使用 Lookahead 優化器更新模型，最後回傳 (final_loss, metrics)
    其中 metrics 包含 slow_similarity 與 final_similarity（此處兩者相同，因無攻擊階段）。
    """
    epochs = config.get("epochs", 5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
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
            print(f"[Lookahead Train] Epoch {epoch+1}/{epochs} completed.")
    
    final_loss, _ = test(net, trainloader)
    return final_loss

# ------------------ Lookahead UPA 訓練 ------------------
def UPA_train(net, trainloader, config, attack_steps: int = 9, attack_alpha: float = 100.0, verbose=False):
    final_loss = train(net, trainloader, config, verbose=False)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 攻擊階段：對單一 batch 執行 UPA 攻擊（不使用 Lookahead）
    net.train()
    try:
        batch = next(iter(trainloader))
    except StopIteration:
        final_loss, _ = test(net, trainloader)
        return final_loss
    
    images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
    images = images.to(DEVICE, memory_format=torch.channels_last)
    
    # 使用單獨的攻擊優化器（不更新 slow weights）
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
    return final_loss

# ------------------ Lookahead TPA 訓練 ------------------
def TPA_train(net, trainloader, config, original_label: int = 0, target_label: int = 2, poison_num: int = 20, attack_steps: int = 9, attack_alpha: float = 1.0, verbose=False):
    final_loss = train(net, trainloader, config, verbose=False)
    
    # 定義損失函數
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
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
        return final_loss
    
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
    return final_loss
