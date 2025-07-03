
from typing import List, Tuple, Optional

import flwr
from flwr.common import EvaluateRes 
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from koshino_FL.koshino_history import history

# disable_progress_bar()

class MyFedAvg(FedAvg): 
    """自訂 FedAvg，以便在 aggregate_evaluate 中記錄分散式指標到 history."""

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        

        # 將每個 client 回傳的 metrics 記錄到 history
        for client_proxy, evaluate_res in results:
            metrics = evaluate_res.metrics  # e.g. {"loss": 0.12, "accuracy": 0.87, ...}
            # print(metrics)
            if "partition_id" in metrics:
                pid = metrics["partition_id"]
            if "client_loss" in metrics:
                loss_val = metrics["client_loss"]
                # 將這個 loss 記錄到分散式 (distributed) 的 loss
                history.add_loss_distributed(server_round, loss_val, pid=pid)
            if "client_accuracy" in metrics:
                acc_val = metrics["client_accuracy"]
                # 將這個 accuracy 記錄到分散式 (distributed) 的 metrics
                history.add_metrics_distributed(server_round, {"accuracy": acc_val}, pid=pid)
        
        # 先呼叫父類別，取得預設的聚合結果(通常是平均loss)
        aggregated_loss = super().aggregate_evaluate(server_round, results, failures)
        return aggregated_loss