import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(history, name='median2', is_per=False):
    """
    獨立的繪製函式，用於顯示集中式與分散式客戶端的準確率隨輪次變化，包含個性化準確率。

    參數:
        history: 帶有 metrics_centralized 和 metrics_distributed 結構的記錄物件。
        name: 用於設定圖表標題和輸出檔案名稱（不含副檔名）。
    """
    # 收集集中式準確率
    rounds_central = [r for r, _ in history.metrics_centralized.get('accuracy', [])]
    acc_central = [a for _, a in history.metrics_centralized.get('accuracy', [])]

    # 收集每個客戶端的分散式全局準確率
    client_acc = {}
    for round_num, pid, acc in history.metrics_distributed.get('accuracy', []):
        client_acc.setdefault(pid, []).append((round_num, acc))

    # 收集每個客戶端的個性化準確率
    client_pers_acc = {}
    for round_num, pid, acc in history.metrics_distributed.get('personalized_accuracy', []):
        client_pers_acc.setdefault(pid, []).append((round_num, acc))

    # 繪製準確率圖表
    plt.figure(figsize=(12, 8))
    plt.plot(rounds_central, acc_central, marker='o', label='Centralized Accuracy', linewidth=2)

    # 為不同客戶端使用不同顏色與樣式
    colors = plt.cm.tab20(np.linspace(0, 1, len(client_acc)))
    for idx, (pid, data) in enumerate(sorted(client_acc.items())):
        rounds, accs = zip(*sorted(data))
        plt.plot(rounds, accs, marker='o', linestyle=':', label=f'Client {pid}', color=colors[idx], alpha=0.7)

        # 繪製個性化準確率（使用相同顏色，但不同樣式）
        if pid in client_pers_acc and is_per == True:
            pers_data = sorted(client_pers_acc[pid])
            pers_rounds, pers_accs = zip(*pers_data)
            plt.plot(pers_rounds, pers_accs, marker='x', linestyle='--', label=f'Client {pid} (Personalized)', color=colors[idx], alpha=0.7)

    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title(name)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # 儲存圖表
    output_path = f"{name}.png"
    plt.savefig(output_path)
    plt.close()


def plot_loss_metrics(history, name='median2_loss', is_per=False):
    """
    繪製每個客戶端的全局損失和個性化損失隨輪次變化的圖表。

    參數:
        history: 帶有 loss_distributed 和 metrics_distributed 結構的記錄物件。
        name: 用於設定圖表標題和輸出檔案名稱（不含副檔名）。
    """
    # 收集每個客戶端的全局損失
    client_loss = {}
    for round_num, pid, loss in history.losses_distributed:
        client_loss.setdefault(pid, []).append((round_num, loss))

    # 收集每個客戶端的個性化損失
    client_pers_loss = {}
    for round_num, pid, metrics in history.metrics_distributed.get('personalized_loss', []):
        client_pers_loss.setdefault(pid, []).append((round_num, metrics))

    # 檢查數據是否為空
    if not client_loss:
        print("警告：未找到任何全局損失數據，無法繪製損失圖表。")
        return
    if not client_pers_loss:
        print("警告：未找到任何個性化損失數據，將只繪製全局損失。")

    # 繪製損失圖表
    plt.figure(figsize=(12, 8))

    # 為不同客戶端使用不同顏色
    colors = plt.cm.tab20(np.linspace(0, 1, len(client_loss)))
    for idx, (pid, data) in enumerate(sorted(client_loss.items())):
        rounds, losses = zip(*sorted(data))
        plt.plot(rounds, losses, marker='o', linestyle=':', label=f'Client {pid} Loss', color=colors[idx], alpha=0.7)

        # 繪製個性化損失（使用相同顏色，但不同樣式）
        if pid in client_pers_loss and is_per == True:
            pers_data = sorted(client_pers_loss[pid])
            pers_rounds, pers_losses = zip(*pers_data)
            plt.plot(pers_rounds, pers_losses, marker='x', linestyle='--', label=f'Client {pid} Loss (Personalized)', color=colors[idx], alpha=0.7)

    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title(f'{name} - Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # 儲存圖表
    output_path = f"{name}.png"
    plt.savefig(output_path)
    plt.close()
