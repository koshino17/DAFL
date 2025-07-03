import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler  # ✅ AMP 混合精度

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#original trainnig
def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.85, weight_decay=5e-4)
    # T_max 建議使用 (NUM_ROUNDS * epochs) 或 (epochs * len(trainloader)) 也可嘗試
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs * len(trainloader)))
    scaler = GradScaler("cuda")  # ✅ AMP 需要 GradScaler 來防止數值溢出
    
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            images = images.to(DEVICE, memory_format=torch.channels_last)
            optimizer.zero_grad()
            
            # ✅ 使用 AMP 自動混合精度
            with autocast("cuda"):
                outputs = net(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()  # ✅ 更新梯度
            
            # Metrics
            epoch_loss += loss.item()
            total += labels.size(0)
            # correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
        
        scheduler.step()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total

        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}")

#UPA trainnig
def UPA_train(
    net, 
    trainloader, 
    epochs: int, 
    alpha: float = 10000.0,   # 惡意攻擊的強度
    verbose=False
):
    #     """
    # 執行'正常訓練' + '惡意操作'的示範函式。
    # - 正常訓練: 與原 train() 相同 (CrossEntropyLoss + label smoothing)
    # - 惡意操作: 在最後，針對一個批次資料進行梯度上升 (gradient ascent)，
    #             使模型參數往"增大損失"的方向更新，破壞全域模型效能。

    # alpha: 攻擊強度係數 (類似學習率)，越大表示往負面方向更新越劇烈。
    # """
    # 這裡假設一些超參數已在外部定義，如 NUM_ROUNDS, DEVICE 等
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs * len(trainloader)))
    scaler = GradScaler("cuda")  # ✅ AMP 需要 GradScaler
    
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            images = images.to(DEVICE, memory_format=torch.channels_last)
            
            optimizer.zero_grad()
            
            # ✅ 使用 AMP 自動混合精度
            with autocast("cuda"):
                outputs = net(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 統計訓練過程
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
        
        scheduler.step()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total

        if verbose:
            print(f"[Malicious] Epoch {epoch+1}: train loss {epoch_loss:.4f}, acc {epoch_acc:.4f}")

    # -----------------------------
    # 這裡開始做「惡意操作」(示範)
    # -----------------------------
    # 1) 取一個批次資料（或多批），再做一次反向傳播，但這次是"梯度上升"。
    #    目標：增加該批資料的損失，以破壞整體模型效能。
    net.train()
    try:
        # 從 trainloader 拿一批資料
        batch = next(iter(trainloader))
    except StopIteration:
        # 如果 trainloader 沒資料，可跳過
        return
    
    images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
    images = images.to(DEVICE, memory_format=torch.channels_last)
    
    # 我們可以使用同一個 optimizer，但要做 "梯度上升"
    # 方式：loss 取負值，然後正常 backward() 即可
    optimizer.zero_grad()
    
    with autocast("cuda"):
        outputs = net(images)
        loss = criterion(outputs, labels)
    
    # 做「負的 loss」反向傳播 => 梯度上升
    neg_loss = -alpha * loss
    scaler.scale(neg_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if verbose:
        print(f"[Malicious Attack] Performed gradient ASCENT with alpha={alpha}, " 
              f"loss on malicious batch={loss.item():.4f}")

#TPA trainnig
def TPA_train(
    net, 
    trainloader, 
    epochs: int,
    original_label: int = 0,  # 想要攻擊的原標籤
    target_label: int = 2,    # 想要把這些原標籤誤分類成什麼
    poison_num: int = 20,     # 從 loader 中抽多少筆屬於 original_label 的樣本來做攻擊
    alpha: float = 1.0,       # 攻擊強度
    verbose=False
):
    """
    在同一個 train 函式中，先做正常訓練 epochs 輪，
    然後自動從 trainloader 中找出指定 original_label 的樣本(至多 poison_num 筆)，
    在最後額外做一次 "Targeted Poisoning Attack" 更新：
      - 把這些樣本的標籤改成 target_label
      - 進行 alpha 倍強度的梯度下降
    """

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs * len(trainloader)))
    scaler = GradScaler("cuda")

    net.train()
    # -----------------------------
    # 1) 正常本地訓練
    # -----------------------------
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch["img"], batch["label"]
            images = images.to(DEVICE, memory_format=torch.channels_last)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = net(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 統計訓練中的 accuracy
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        scheduler.step()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total

        if verbose:
            print(f"[Train] Epoch {epoch+1}/{epochs}, loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

    # -----------------------------
    # 2) 目標式中毒攻擊 (Targeted Poisoning) - inline
    # -----------------------------
    net.train()

    # 先從 trainloader 裡面，再「額外」取出 up to `poison_num` 筆樣本，label=original_label
    poison_imgs = []
    poison_labels = []
    count = 0

    # 這裡重新遍歷一次 trainloader，也可以事先做 snapshot
    for batch in trainloader:
        imgs, lbs = batch["img"], batch["label"]
        imgs = imgs.to(DEVICE, memory_format=torch.channels_last)
        lbs = lbs.to(DEVICE)

        for i in range(len(imgs)):
            if lbs[i].item() == original_label:
                poison_imgs.append(imgs[i])
                poison_labels.append(lbs[i])
                count += 1
                if count >= poison_num:
                    break
        if count >= poison_num:
            break

    # 如果沒找到任何樣本，就不用做攻擊
    if len(poison_imgs) == 0:
        if verbose:
            print(f"[TPA] No sample found for label={original_label}, skip targeted attack.")
        return

    # 轉成 Tensor
    poison_imgs = torch.stack(poison_imgs, dim=0)   # shape: [N, C, H, W]
    poison_labels = torch.stack(poison_labels, dim=0)  # shape: [N]
    
    # 定義假標籤
    fake_labels = torch.full_like(poison_labels, target_label).to(DEVICE)

    # 攻擊步驟(可做多次), 這裡示範 1 次
    steps = 1
    for s in range(steps):
        optimizer.zero_grad()
        with autocast("cuda"):
            outputs = net(poison_imgs)
            # 將 loss 乘上 alpha (放大攻擊效果)
            loss = criterion(outputs, fake_labels) * alpha

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if verbose:
        print(f"[TPA] Attack done on label={original_label} -> {target_label}, final poison_loss={loss.item():.4f}")