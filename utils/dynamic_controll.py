import random
import threading
from typing import List, Callable
from utils.clientmanger import DynamicClientManager


def base_on_off(
    client_manager: DynamicClientManager,
    current_round,
    online_list: List[str],
    offline_list: List[str],
    DFL: bool
):
    # 下線部分在線客戶端        
    if current_round<4 and online_list and DFL:
        offline_num = 1
        for cid in random.sample(online_list, offline_num):
            client = client_manager.clients.get(cid)
            if client:
                client_manager.unregister(client)

    # 上線部分離線客戶端
    if current_round>3 and offline_list and DFL:
        online_num = 1
        for cid in random.sample(offline_list, online_num):
            client = client_manager.off_clients.get(cid)
            if client:
                print(f"[TOGGLER] 將 {cid} 上線")
                client_manager.register(client)

def random_on_off(
    client_manager: DynamicClientManager,
    current_round,
    online_list: List[str],
    offline_list: List[str],
    offline_rate: float,
    online_rate: float,
    DFL: bool 
):
    try:
        # 確保列表不為空
        # if not online_list or not offline_list:
        #     print(f"警告: 客戶端列表為空 (在線:{len(online_list)}, 離線:{len(offline_list)})")
        print("user online:", len(online_list), "offline:", len(offline_list))
        
        if current_round == 1 and online_list and DFL:
            # 在第一輪隨機選擇一個在線客戶端下線
            total_clients = len(online_list) + len(offline_list)
            offline_num = max(1, int(offline_rate * total_clients))
            offline_num = min(offline_num, len(online_list))  # 確保不超過在線客戶端數量
            if offline_num > 0:
                for cid in random.sample(online_list, offline_num):
                    client = client_manager.clients.get(cid)
                    if client:
                        client_manager.unregister(client)
        
        # 下線部分在線客戶端
        if current_round % 2 != 0 and current_round != 1 and online_list and DFL:
            # 確保取樣數量不超過列表長度
            total_clients = len(online_list) + len(offline_list)
            offline_num = max(1, int(offline_rate * total_clients))
            offline_num = min(offline_num, len(online_list))  # 確保不超過在線客戶端數量
            if offline_num > 0:
                for cid in random.sample(online_list, offline_num):
                    client = client_manager.clients.get(cid)
                    if client:
                        client_manager.unregister(client)
        
        # 上線部分離線客戶端
        if current_round % 2 == 0 and offline_list and DFL:
            # 確保取樣數量不超過列表長度
            online_num = min(1, len(offline_list))
            if online_num > 0:
                for cid in random.sample(offline_list, online_num):
                    client = client_manager.off_clients.get(cid)
                    if client:
                        print(f"[TOGGLER] 將 {cid} 上線")
                        client_manager.register(client)
    except Exception as e:
        print(f"random_on_off 執行錯誤: {e}")
        import traceback
        traceback.print_exc()

def background_online_offline_simulator(
    client_manager: DynamicClientManager,
    get_round: Callable[[], int],
    event: threading.Event,
    # interval: int = 30,
    offline_rate: float,
    online_rate: float,
    DFL: bool
):
    while True:
        event.wait()  # 等待 server 通知新的一輪開始
        # time.sleep(interval)
        current_round = get_round()
        
        online_list = list(client_manager.online_clients)
        offline_list = list(client_manager.offline_clients)

        # base_on_off(client_manager, current_round, online_list, offline_list, DFL)
        random_on_off(client_manager, current_round, online_list, offline_list, offline_rate, online_rate, DFL)

        event.clear()  # 清除標記，等待下一次 signal
