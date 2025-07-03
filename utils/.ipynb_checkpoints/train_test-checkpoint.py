from collections import OrderedDict
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler  # 使用 torch.cuda.amp 模組
from utils.Lookahead import Lookahead
from utils.others import get_parameters
from torch.optim.lr_scheduler import CosineAnnealingLR
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
def train(net, trainloader, parameters, config, partition_id, verbose=False):
    expected_params = len(net.state_dict().keys())
    if len(parameters) != expected_params:
        print(f"客戶端 {partition_id}: 參數數量不匹配! 期望 {expected_params}, 得到 {len(parameters)}")
        # 即使參數不匹配，仍確保返回正確格式
        return get_parameters(config), len(trainloader.dataset), {"certainty": 1.0}
    else:
        state_dict = OrderedDict()
        for (name, _), param in zip(net.state_dict().items(), parameters):
            # 確保轉換為浮點張量
            state_dict[name] = torch.tensor(param, dtype=torch.float, device=DEVICE)
        net.load_state_dict(state_dict)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    criterion = nn.CrossEntropyLoss()
    # 獲取梯度
    current_loss, grad = get_grad(net, trainloader, DEVICE, partition_id)
    
    # 執行本地訓練
    for epoch in range(config["local_epochs"]):
        epoch_loss = 0.0
        batch_count = 0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        avg_epoch_loss = epoch_loss / max(1, batch_count)
        # print(f"客戶端 {partition_id}, Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}")
    
    # 建議添加返回值，使功能完整
    return grad, current_loss

# ------------------ Lookahead UPA 訓練 ------------------
def lookahead_UPA_train(net, trainloader, parameters, config, partition_id, attack_steps: int = 3, attack_alpha: float = 4.0, verbose=False):
    """
    使用 Lookahead 進行正常訓練，再對單一 batch 執行 UPA 攻擊（梯度上升）。
    回傳 final_loss。
    """
    # 執行正常訓練
    grad, current_loss = train(net, trainloader, parameters, config, partition_id, verbose=False)

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
    return grad, final_loss

# ------------------ Lookahead TPA 訓練 ------------------
def lookahead_TPA_train(net, trainloader, parameters, config, partition_id, original_label: int = 0, target_label: int = 2,
                        poison_num: int = 20, attack_steps: int = 3, attack_alpha: float = 4.0, verbose=False):
    """
    使用 Lookahead 進行正常訓練，再從 trainloader 中挑出指定 original_label 的樣本，
    將其標籤改為 target_label，並進行多步驟梯度下降攻擊 (targeted poisoning)。
    回傳 final_loss。
    """
    # 執行正常訓練
    grad, current_loss = train(net, trainloader, parameters, config, partition_id, verbose=False)

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
        return grad, current_loss

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
    return grad, final_loss

def local_evaluate(parameters, net, device, valloader, partition_id):
    expected_params = len(net.state_dict().keys())
    if len(parameters) != expected_params:
        print(f"客戶端 {partition_id}: 評估參數數量不匹配! 期望 {expected_params}, 得到 {len(parameters)}")
    else:
        state_dict = OrderedDict()
        for (name, _), param in zip(net.state_dict().items(), parameters):
            state_dict[name] = torch.tensor(param, device=device)  # 使用傳入的 device 參數
        net.load_state_dict(state_dict)
    
    net.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    accum_error = 0
    with torch.no_grad():
        for batch in valloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = net(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            accum_error += criterion(outputs, labels).item() * labels.size(0)
    
    accuracy = correct / total
    avg_loss = accum_error / total
    
    return float(avg_loss), float(accuracy)  # 添加返回值，使函數更完整

def get_grad(net, trainloader, device, partition_id):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    try:
        batch = next(iter(trainloader))
        images, labels = batch["img"].to(device), batch["label"].to(device)
        net.zero_grad()  # Clear any existing gradients
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        grad = OrderedDict()
        for name, param in net.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad.clone()
            else:
                grad[name] = torch.zeros_like(param)
        return loss.item(), grad
    except StopIteration:
        print(f"客戶端 {partition_id}: 訓練數據迭代器無效")
        return 0.0, OrderedDict()
