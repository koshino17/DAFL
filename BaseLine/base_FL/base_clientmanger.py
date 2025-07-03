
from typing import List, Optional

import random

from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy

class DynamicClientManager(SimpleClientManager):
    def __init__(self):
        super().__init__()
        self.online_clients = set()
        self.offline_clients = set()  # 新增離線列表
        self.off_clients: dict[str, ClientProxy] = {}  # 正確初始化為空字典

    def register(self, client: ClientProxy) -> bool:
        if client.cid in self.offline_clients:
            self.offline_clients.remove(client.cid)
        self.online_clients.add(client.cid)
        print(f"[Status] Client {client.cid} 已上線 ✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓")
        return super().register(client)

    def unregister(self, client: ClientProxy) -> None:
        if client.cid in self.online_clients:
            self.off_clients[client.cid] = client
            self.online_clients.remove(client.cid)
            self.offline_clients.add(client.cid)  # 移至離線列表
            print(f"[Status] Client {client.cid} 已下線 ✗")
        super().unregister(client)

    def sample(self, num_clients: int, min_num_clients: Optional[int] = None) -> List[ClientProxy]:
        available = [cid for cid in self.online_clients if cid in self.clients]
        num_available = len(available)
        if num_available == 0:
            return []
        n_sample = max(num_clients, min_num_clients or 0)
        sampled_cids = random.sample(available, min(n_sample, num_available))
        return [self.clients[cid] for cid in sampled_cids]