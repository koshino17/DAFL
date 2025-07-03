# koshino_FL/koshino_history.py
from flwr.server.history import History

# history = History()

class CustomHistory(History):
    def add_loss_distributed(self, round_num, loss, pid=None):
        # 使用 pid 代替 client_id
        self.losses_distributed.append((round_num, pid, loss))

    def add_metrics_distributed(self, round_num, metrics, pid=None):
        # 使用 pid 代替 client_id
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_distributed:
                self.metrics_distributed[metric_name] = []
            self.metrics_distributed[metric_name].append((round_num, pid, value))

history = CustomHistory()