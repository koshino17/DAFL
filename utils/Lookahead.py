from typing import List
import numpy as np
import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer

class Lookahead(Optimizer):
    r"""實作 Lookahead 優化器，包裝任意 PyTorch 基礎優化器。

    每 k 步後，更新慢速權重：
      slow = slow + α * (fast - slow)
      fast <- slow

    參數：
        optimizer (Optimizer): 基礎優化器（例如 SGD、Adam 等）。
        alpha (float, optional): 慢速權重的更新步長（0 到 1 之間），預設為 0.5。
        k (int, optional): 更新慢速權重的步數，預設為 5。
    """
    def __init__(self, optimizer, alpha=0.5, k=5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Invalid alpha parameter: {}. It must be in [0, 1].".format(alpha))
        if k < 1:
            raise ValueError("Invalid k parameter: {}. It must be at least 1.".format(k))
        self.optimizer = optimizer
        self.alpha = alpha
        self.k = k
        self.step_counter = 0
        # 為每個參數建立慢速權重副本
        self.slow_weights = []
        for group in optimizer.param_groups:
            slow_group = []
            for p in group['params']:
                slow_group.append(p.clone().detach())
            self.slow_weights.append(slow_group)

    def step(self, closure=None):
        """執行一次 Lookahead 更新步驟。"""
        loss = self.optimizer.step(closure)
        self.step_counter += 1
        if self.step_counter % self.k == 0:
            # 更新所有參數組的慢速權重，並同步到快權重
            for group_idx, group in enumerate(self.optimizer.param_groups):
                for p_idx, p in enumerate(group['params']):
                    slow = self.slow_weights[group_idx][p_idx]
                    # 更新公式：slow = slow + alpha * (p - slow)
                    slow.data.add_(p.data - slow.data, alpha=self.alpha)
                    # 同步更新：快權重更新為慢速權重
                    p.data.copy_(slow.data)
        return loss

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        # 使 Lookahead 可以正確地暴露內部優化器的參數組，供 lr_scheduler 使用
        return self.optimizer.param_groups

    def get_slow_weights(self) -> List[np.ndarray]:
        """回傳慢速權重的數值版（以 numpy 陣列形式）"""
        slow_params = []
        for group in self.slow_weights:
            for p in group:
                slow_params.append(p.cpu().detach().numpy())
        return slow_params
