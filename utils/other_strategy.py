
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import flwr
from flwr.common import EvaluateRes 
from flwr.server.strategy import FedAvg, FedAdam
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import NDArrays, Parameters
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from collections import defaultdict, OrderedDict

from utils.history import history
from utils.weights_utils import weights_substraction
from utils.optim import AdaAdam
# disable_progress_bar()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    
class AdaFedAdamStrategy(FedAvg):
    def __init__(
        self,
        net: nn.Module,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        eta: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        tau: float = 1e-8,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            # eta=eta,
            # eta_l=eta,
            # beta_1=beta_1,
            # beta_2=beta_2,
            # tau=tau,
        )
        self.eta = eta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tau = tau
        self.model = net.to(DEVICE)
        self.optimizer = AdaAdam(
            params=self.model.parameters(),
            lr=eta,
            betas=(beta_1, beta_2),
            eps=tau,
        )
        self.previous_weights = None

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is None:
            return None, {}
        nds = parameters_to_ndarrays(parameters)
        
        expected_count = len(self.model.state_dict().keys())
        if len(nds) != expected_count:
            print(f"警告: 參數數量不匹配! 期望 {expected_count}, 得到 {len(nds)}")
            fresh_model = self.model
            fresh_params = [param.cpu().numpy() for param in fresh_model.state_dict().values()]
            print(f"重置為新模型參數 (長度 {len(fresh_params)})")
            return ndarrays_to_parameters(fresh_params), metrics
        
        device = next(self.model.parameters()).device
        state_dict = OrderedDict()
        for (name, _), nd in zip(self.model.state_dict().items(), nds):
            # 確保轉換為浮點張量
            state_dict[name] = torch.tensor(nd, dtype=torch.float, device=device)
        self.model.load_state_dict(state_dict)
        
        current_params = {name: param.detach().clone().float() for name, param in self.model.named_parameters()}
        
        if self.previous_weights is None:
            self.previous_weights = current_params
            return ndarrays_to_parameters([param.cpu().numpy() for param in self.model.state_dict().values()]), metrics
        
        
        # Aggregate updates and certainty from clients
        total_weight = 0
        certainty_sum = 0
        pseudo_grad = OrderedDict()
        
        for _, fit_res in results:
            if "certainty" not in fit_res.metrics:
                print(f"警告: 客戶端未返回certainty，跳過該客戶端")
                continue
            
            params = parameters_to_ndarrays(fit_res.parameters)
            client_params = OrderedDict()
            for (name, _), param in zip(self.model.state_dict().items(), params):
                client_params[name] = torch.tensor(param, device=device)
            
            # Compute update (difference between previous and current client params)
            update = weights_substraction(self.previous_weights, client_params)
            
            # Use the client-provided certainty and weight
            weight = fit_res.num_examples  # Using num_examples as weight
            certainty = fit_res.metrics["certainty"]
            
            total_weight += weight
            certainty_sum += certainty * weight
            
            for key in update:
                if key not in pseudo_grad:
                    pseudo_grad[key] = weight * update[key]
                else:
                    pseudo_grad[key] += weight * update[key]
        
        if total_weight <= 0:
            print(f"警告: 總權重為0，無法聚合")
            return ndarrays_to_parameters([param.cpu().numpy() for param in self.model.state_dict().values()]), metrics
        
        certainty = certainty_sum / total_weight
        for key in pseudo_grad:
            pseudo_grad[key] = pseudo_grad[key] / total_weight
        
        self.optimizer.zero_grad()
        for k, w in self.model.named_parameters():
            if w.requires_grad and k in pseudo_grad:
                w.grad = pseudo_grad[k]
        
        # print(f"第 {server_round} 輪客戶端確定性：{certainty:.4f}")
        self.optimizer.set_confidence(certainty)
        self.optimizer.step()
        
        final_state_dict = self.model.state_dict()
        updated_params = [param.cpu().numpy() for param in final_state_dict.values()]
        self.previous_weights = {name: param.detach().clone() for name, param in self.model.named_parameters()}
        
        return ndarrays_to_parameters(updated_params), metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        for client_proxy, evaluate_res in results:
            metrics = evaluate_res.metrics
            if "partition_id" in metrics:
                pid = int(metrics["partition_id"])
                if "client_accuracy" in metrics:
                    acc_val = metrics["client_accuracy"]
                    history.add_metrics_distributed(server_round, {"accuracy": acc_val}, pid=pid)
                if "client_loss" in metrics:
                    loss_val = metrics["client_loss"]
                    history.add_loss_distributed(server_round, loss_val, pid=pid)
        
        return super().aggregate_evaluate(server_round, results, failures)

