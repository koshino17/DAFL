from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import flwr
from flwr.common import EvaluateRes, FitRes, EvaluateIns, Parameters, Scalar, FitIns
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from collections import OrderedDict
import pandas as pd
from IPython.display import display
from sklearn.cluster import DBSCAN
import math
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Subset, DataLoader
from utils.history import history
from utils.other_strategy import MyFedAvg
from utils.time import timed
from utils.optim import AdaAdam
from utils.Similarity_Measurement import _compute_similarity_compressed
from utils.others import check_and_clean_params, set_parameters, init_or_check_client_info_table, record_metrics
from utils.train_test import test
from utils.aggregation import median_aggregation, non_IID_aggregation
from utils.loaddata import get_cached_datasets

import cProfile
import pstats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Q = 0.9
quick_mode = False  # 是否使用快速模式（只用於測試）

class HRFA(MyFedAvg):
    """
    在 MyFedAvg 基礎上，加入『動態客戶端管理』的功能，並使用 DataFrame 儲存客戶端資訊。
    """
    def __init__(self,
                 fraction_fit,
                 net,
                 alpha=0.5,     
                 eta: float = 1e-3,
                beta_1: float = 0.9,
                beta_2: float = 0.999,
                tau: float = 1e-8,
                 **kwargs):
        super().__init__(**kwargs)
        self.fraction_fit = fraction_fit  # 想要每回合選擇多少比例的客戶端
        self.alpha = alpha  # 用於滑動平均
        self.client_info_table: Dict[str, Dict[str, float]] = {}  # 記錄客戶端資訊
        self.client_info_df = pd.DataFrame()  # 存儲客戶端資訊的 DataFrame
        self.previous_global_params = None  # 用來保存上一輪分發給客戶端的全局參數
        self.metric_history: Dict[str, Dict[str, List[float]]] = {}
        self.net = net
        self.optimizer = AdaAdam(
            params=self.net.parameters(),
            lr=eta,
            betas=(beta_1, beta_2),
            eps=tau,
        )
        self.previous_weights = None

    # @timeit_step("configure_fit")
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: flwr.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # 在每輪開始時先做時間衰減
        apply_time_decay(self.client_info_table, decay_factor=0.95)
        # 記錄當前回合發出的全局參數（用來做 similarity 比較）
        self.previous_global_params = parameters_to_ndarrays(parameters)
        old_fraction = self.fraction_fit  # 暫存舊值

        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        fit_ins = FitIns(parameters, config)
    
        all_clients = list(client_manager.all().values())
        # 針對所有用戶端id做初始化，避免 KeyError
        client_ids = [client.cid for client in all_clients]
        init_or_check_client_info_table(self.client_info_table, client_ids)
        
        if server_round <= 1:
            self.fraction_fit = 1.0
            fit_ins = super().configure_fit(server_round, parameters, client_manager)
            # 初始化表格資訊（若該client還沒有記錄）
            all_client_ids = [c[0].cid for c in fit_ins]
            init_or_check_client_info_table(self.client_info_table, all_client_ids)
            self.fraction_fit = old_fraction
            return fit_ins
        
        # 先依照 cid 排序，確保結果固定
        sorted_clients = sorted(all_clients,
                                key=lambda x: self.client_info_table[x.cid]["reputation"],  # 改用長期RS排序
                                reverse=True,)
        
        # 計算要選取的客戶端數量 (例如10個客戶端，fraction_fit=0.5，則選取 int(10*0.5) = 5 個)
        num_clients_to_select = int(len(sorted_clients) * self.fraction_fit)
        
        # 選取前面 num_clients_to_select 個客戶端
        selected_clients = sorted_clients[:num_clients_to_select]
        
        # 回傳選取的客戶端與 FitIns
        return [(client, fit_ins) for client in selected_clients]

    # @timeit_step("aggregate_fit")
    def aggregate_fit(
        self,
        server_round: int,
        fit_results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
    
    	# 1) 收集所有 client 的參數
        client_params_list = {
            client_proxy.cid: parameters_to_ndarrays(fit_res.parameters)
            for client_proxy, fit_res in fit_results
        }

        if not quick_mode:
        # 2) 更新 client_info_table
            update_client_info_table(self.client_info_table, fit_results, self.previous_global_params, self.net, alpha=self.alpha)

            record_metrics(self.metric_history, self.client_info_table)

        aggregated_ndarrays = median_aggregation(client_params_list)
	
        # 5) 更新 self.previous_global_params
        self.previous_global_params = aggregated_ndarrays
        
        
        updated_params = non_IID_aggregation(self.net, self.optimizer, fit_results, aggregated_ndarrays, self.previous_weights)
        self.previous_weights = {name: param.detach().clone() for name, param in self.net.named_parameters()}
	
        # # 6) 轉回成 Flower 需要的型態 (Parameters)
        aggregated_parameters = ndarrays_to_parameters(updated_params)

        # 7) 回傳
        return aggregated_parameters, {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        
        losses = []
        accuracies = []
        # 將每個 client 回傳的 metrics 記錄到 history
        for client_proxy, evaluate_res in results:
            cid = client_proxy.cid
            if cid not in self.client_info_table:
                init_or_check_client_info_table(self.client_info_table, [cid])

            metrics = evaluate_res.metrics  # e.g. {"loss": 0.12, "accuracy": 0.87, ...}
            # print(metrics)
            if "partition_id" in metrics:
                pid = metrics["partition_id"]
            if "client_loss" in metrics:
                loss_val = metrics["client_loss"]
                losses.append(loss_val)
                self.client_info_table[cid]["client_loss"] = loss_val
                # 將這個 loss 記錄到分散式 (distributed) 的 loss
                history.add_loss_distributed(server_round, loss_val, pid=pid)
                # 維護 loss_history
                loss_history = self.client_info_table[cid].get("loss_history", [])
                if loss_val < 9999.9:
                    loss_history.append(loss_val)
                if len(loss_history) > 5:
                    loss_history.pop(0)
                self.client_info_table[cid]["loss_history"] = loss_history

            if "client_accuracy" in metrics:
                acc_val = metrics["client_accuracy"]
                accuracies.append(acc_val)
                self.client_info_table[cid]["client_accuracy"] = acc_val
                # 將這個 accuracy 記錄到分散式 (distributed) 的 metrics
                history.add_metrics_distributed(server_round, {"accuracy": acc_val}, pid=pid)
                # 維護 accuracy_history
                accuracy_history = self.client_info_table[cid].get("accuracy_history", [])
                accuracy_history.append(acc_val)
                if len(accuracy_history) > 10:
                    accuracy_history.pop(0)
                self.client_info_table[cid]["accuracy_history"] = accuracy_history
            
        
        # 先呼叫父類別，取得預設的聚合結果(通常是平均loss)
        aggregated_loss = super().aggregate_evaluate(server_round, results, failures)
        aggregated_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        
        return aggregated_loss, {"accuracy": aggregated_accuracy}


def check_and_clean_params(
    params: List[np.ndarray],
    cid: str
) -> List[np.ndarray]:
    """若參數含有 NaN/Inf，做一些簡單處理（例如直接設為0）."""
    cleaned_params = []
    for idx, w in enumerate(params):
        if np.isnan(w).any() or np.isinf(w).any():
            print(f"[Warning] Client {cid}, layer {idx} has NaN/Inf - replacing with zeros.")
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        cleaned_params.append(w)
    return cleaned_params

def init_or_check_client_info_table(
    client_info_table: Dict[str, Dict[str, float]],
    client_ids: List[str],
) -> None:
    """
    檢查 client_info_table 是否已有對應的 client id，
    若沒有則初始化該 client 在表裡的紀錄。
    """
    for cid in client_ids:
        if cid not in client_info_table:
            # 初始化 reputation, last_loss 等等
            client_info_table[cid] = {
                "partition_id": -1,
                "client_accuracy": 0.0,
                "is_suspicious": False,
                "reputation": 0.3,        # 綜合聲譽（可改為長期聲譽）
                "short_term_rs": 0.3,     # 新增：短期RS
                "long_term_rs": 0.3,      # 新增：長期RS
                "client_loss": 9999.9,
                "loss_history": [],       # 新增：用於計算波動度
                "similarity": -1.0,
                "similarity_history": [],  # 新增：用於長期可靠性
                "shapley": 0.0,
                "shapley_history": [],
            }

def re_aggregate_without(
    client_params_dict: Dict[str, List[np.ndarray]],
    exclude_cid: str,
    client_info_table: Dict[str, Dict[str, float]],
    beta: float = 0.7,
) -> Optional[List[np.ndarray]]:
    """
    重新聚合(加權)所有 client 裡，排除 exclude_cid 的參數。
    回傳該「去掉某人」後的聚合模型參數。
    """
    # 1. 過濾掉想要排除的 client
    if exclude_cid is None:
        # 不排除任何客戶端
        partial_clients = client_params_dict.copy()
    else:
        # 排除指定的客戶端
        partial_clients = {
            cid: params for cid, params in client_params_dict.items() 
            if cid != exclude_cid
        }
    if not partial_clients:
        # 意味著只有這個 cid 一個客戶端在傳參數，移除後就空了
        return None
    
    # 2. 做 NaN/Inf 清洗
    for cid, params in partial_clients.items():
        partial_clients[cid] = check_and_clean_params(params, cid)
    
    # 3. 根據 short_term_rs/long_term_rs 做加權聚合
    weighted_params = None
    total_weight = 0.0
    weights = {}
    
    for cid, params in partial_clients.items():
        info = client_info_table.get(cid, {})
        st_rs = info.get("short_term_rs", 0.3)  # 預設值改為 0.3
        lt_rs = info.get("long_term_rs", 0.3)   # 預設值改為 0.3
        
        # 確保聲譽分數不會是 NaN 或 Inf
        st_rs = np.nan_to_num(st_rs, nan=0.3, posinf=1.0, neginf=0.0)
        lt_rs = np.nan_to_num(lt_rs, nan=0.3, posinf=1.0, neginf=0.0)
        
        combined_weight = beta * lt_rs + (1 - beta) * st_rs
        combined_weight = max(combined_weight, 0.1)  # 最小權重為 0.1
        
        weights[cid] = combined_weight
        total_weight += combined_weight
        
        if weighted_params is None:
            weighted_params = [combined_weight * p for p in params]
        else:
            for idx, p in enumerate(params):
                # 檢查並處理 NaN/Inf 值
                if np.isnan(p).any() or np.isinf(p).any():
                    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                weighted_params[idx] += combined_weight * p
    
    if total_weight <= 1e-9:
        # 如果仍然總權重太小，使用均等權重
        print("[Warning] re_aggregate_without: total_weight too small, using equal weights")
        total_weight = len(partial_clients)
        weighted_params = None
        for cid, params in partial_clients.items():
            if weighted_params is None:
                weighted_params = [p.copy() for p in params]
            else:
                for idx, p in enumerate(params):
                    weighted_params[idx] += p
    
    # 最終正規化並檢查 NaN/Inf
    aggregated = []
    for p in weighted_params:
        normalized_p = p / total_weight
        # 確保結果沒有 NaN/Inf
        normalized_p = np.nan_to_num(normalized_p, nan=0.0, posinf=0.0, neginf=0.0)
        aggregated.append(normalized_p)
    return aggregated

def calculate_shapley(
    client_models: Dict[str, List[np.ndarray]],
    net,
    test_loader: DataLoader,
    client_info_table: Dict[str, Dict[str, float]],
    n_permutations: int = 20,
    use_robust_agg: bool = True,
    beta: float = 0.7
) -> Dict[str, float]:
    """
    改進版 Shapley 計算:
    1. 採用蒙特卡洛近似法，隨機採樣 client 排列組合
    2. 結合防禦機制，排除可疑客戶端
    3. 整合多指標 (loss + accuracy)
    4. 加入聲譽權重調整邊際貢獻
    """
    shapley_values = {cid: 0.0 for cid in client_models}
    total_perm = 0
    
    # 預先過濾可疑客戶端
    valid_clients = [cid for cid in client_models if not client_info_table[cid].get("is_suspicious", False)]
    if not valid_clients:
        return {cid: 0.0 for cid in client_models}
    
    # 預先計算全體客戶端聚合模型 (用於基準比較)
    global_model = re_aggregate_without(client_models, exclude_cid=None, 
                                       client_info_table=client_info_table, beta=beta)
    if global_model is None:
        # 如果無法聚合全局模型，回傳空的 Shapley 值
        return {cid: 0.0 for cid in client_models}
    
    # 基準評估 (loss + accuracy)
    base_loss, base_acc = evaluate_model_with_metrics(net, global_model, test_loader)
    
    for _ in range(n_permutations):
        # 隨機生成 client 排列
        perm = np.random.permutation(valid_clients)
        prev_set = []
        
        for idx, cid in enumerate(perm):
            # 計算邊際貢獻: 加入 cid 前後的差異
            current_set = prev_set + [cid]
            
            # 聚合當前集合模型 (加入防禦聚合)
            subset_params = {k: client_models[k] for k in current_set}
            if use_robust_agg:
                agg_model = re_aggregate_without(subset_params, exclude_cid=None,
                                                client_info_table=client_info_table,
                                                beta=beta)
            else:
                agg_model = median_aggregation(subset_params)
            
            # 檢查聚合模型是否為 None
            if agg_model is None:
                # 跳過此次計算，使用預設值
                curr_loss, curr_acc = base_loss, 0.0
            else:
                # 評估當前集合
                curr_loss, curr_acc = evaluate_model_with_metrics(net, agg_model, test_loader)
            
            # 計算邊際貢獻 (整合 loss 和 accuracy)
            if prev_set:
                if use_robust_agg:
                    prev_agg = re_aggregate_without({k: client_models[k] for k in prev_set}, 
                                                  exclude_cid=None, 
                                                  client_info_table=client_info_table,
                                                  beta=beta)
                else:
                    prev_agg = median_aggregation({k: client_models[k] for k in prev_set})
                
                # 檢查前一個聚合模型是否為 None
                if prev_agg is None:
                    prev_loss, prev_acc = base_loss, 0.0
                else:
                    prev_loss, prev_acc = evaluate_model_with_metrics(net, prev_agg, test_loader)
            else:
                prev_loss, prev_acc = base_loss, 0.0  # 空集合表現為基準
                
            # 綜合貢獻值 (可調整權重)
            marginal = (prev_loss - curr_loss) + 0.5*(curr_acc - prev_acc)
            
            # 檢查並處理 marginal 中的 NaN/Inf
            if np.isnan(marginal) or np.isinf(marginal):
                marginal = 0.0
            
            # 加入聲譽權重調整
            rep_weight = client_info_table[cid].get("reputation", 0.3)
            rep_weight = np.nan_to_num(rep_weight, nan=0.3, posinf=1.0, neginf=0.0)
            
            weighted_marginal = marginal * rep_weight
            
            # 最終檢查
            if np.isnan(weighted_marginal) or np.isinf(weighted_marginal):
                weighted_marginal = 0.0
            
            shapley_values[cid] += weighted_marginal
            total_perm += 1
            
            prev_set = current_set.copy()
    
    # 平均化並正規化
    if total_perm > 0:
        for cid in valid_clients:
            shapley_values[cid] /= total_perm
            # 處理 NaN/Inf 值
            shapley_values[cid] = np.nan_to_num(shapley_values[cid], nan=0.0, posinf=1.0, neginf=0.0)
        
        # 安全的正規化
        valid_values = [v for v in shapley_values.values() if not (np.isnan(v) or np.isinf(v))]
        if valid_values:
            max_shap = max(valid_values)
            if max_shap > 1e-9:  # 避免除以接近零的數
                for cid in shapley_values:
                    shapley_values[cid] = shapley_values[cid] / max_shap
                    # 確保在 0~1 範圍內
                    shapley_values[cid] = max(0.0, min(1.0, shapley_values[cid]))
            else:
                # 如果所有值都接近零，設為預設值
                for cid in shapley_values:
                    shapley_values[cid] = 0.1
        else:
            # 如果所有值都是 NaN/Inf，設為預設值
            for cid in shapley_values:
                shapley_values[cid] = 0.1
                
    return shapley_values

def evaluate_model_with_metrics(net, params: List[np.ndarray], test_loader: DataLoader) -> Tuple[float, float]:
    """同時回傳 loss 和 accuracy"""
    set_parameters(net, params)
    loss, accuracy = test(net, test_loader)
    return float(loss), float(accuracy)

def update_client_info_table(
    client_info_table: Dict[str, Dict[str, float]],
    fit_results: List[Tuple[ClientProxy, FitRes]],
    global_model: List[np.ndarray],  # 新增參數
    net,
    alpha: float = 0.5,
) -> None:
    """
    根據本輪的訓練結果 (loss, metrics...) 更新 client_info_table。
    除了更新 loss 與 reputation，也更新 similarity（看作是慢速更新與全局參數間的相似性）。
    """
    # print("[INFO] global_model :", global_model)
    _, _, test_loader = get_cached_datasets(0, q=Q)
    trained_cids = {client_proxy.cid for (client_proxy, _) in fit_results}
    # print('[INFO] trained_cids:', trained_cids)
    trained_partitions = {
        fit_res.metrics.get("partition_id", client_proxy.cid)
        for (client_proxy, fit_res) in fit_results
    }
    print('[INFO] trained_partitions:', trained_partitions)
    
    # (修正) 準備當前 client_models 給 Shapley 計算
    client_models_dict = {}
    for cp, fit_res in fit_results:
        cid = cp.cid
        
        # 清理 NaN/Inf
        raw_params = parameters_to_ndarrays(fit_res.parameters)
        raw_params = check_and_clean_params(raw_params, cid)
        client_models_dict[cid] = raw_params

    if not quick_mode:
        shapley_values = calculate_shapley(
            client_models=client_models_dict,
            net=net,
            test_loader=test_loader,
            client_info_table=client_info_table,
            n_permutations=20,       # 可調整採樣次數
            use_robust_agg=True,     # 是否使用防禦聚合
            beta=0.7
        )
        print("[INFO] shapley_values:", shapley_values)
        
        # 寫入 client_info_table
        for cid, shap_val in shapley_values.items():
            client_info_table[cid]["shapley"] = shap_val
            hist = client_info_table[cid].get("shapley_history", [])
            hist.append(shap_val)
            if len(hist) > 5:
                hist.pop(0)
            client_info_table[cid]["shapley_history"] = hist

    # 3) 更新其餘資訊 (loss, accuracy, similarity, short_term_rs...)
    for (client_proxy, fit_res) in fit_results:
        cid = client_proxy.cid
        fit_metrics = fit_res.metrics
        
        # —— 新增：更新 partition_id —— 
        if "partition_id" in fit_metrics:
            client_info_table[cid]["partition_id"] = fit_metrics["partition_id"]

        # (a) 更新 loss 與 accuracy
        loss_history = client_info_table[cid].get("loss_history", [])
        latest_acc = float(fit_metrics.get("client_accuracy", 0.0))

        # (d) 計算與 global params 的 similarity
        client_params_nd = parameters_to_ndarrays(fit_res.parameters)
        
        if global_model is not None:
            #計算similarity
            similarity = _compute_similarity_compressed(global_model, client_params_nd)
            # print(f"[INFO] Client {cid} similarity: {similarity:.4f}")
            client_info_table[cid]["similarity"] = similarity
            sim_history = client_info_table[cid].get("similarity_history", [])
            sim_history.append(similarity)
            if len(sim_history) > 5:
                sim_history.pop(0)
            client_info_table[cid]["similarity_history"] = sim_history

        # (e) 計算短期RS (與原程式相同)
        loss_std = np.std(loss_history) if len(loss_history) >= 2 else 0
        loss_std = np.nan_to_num(loss_std, nan=0.0, posinf=0.0, neginf=0.0)
        
        similarity_now = client_info_table[cid]["similarity"]
        similarity_now = np.nan_to_num(similarity_now, nan=0.0, posinf=1.0, neginf=0.0)

        short_term_rs = (
            0.6 * latest_acc +
            0.2 * (1 - loss_std) +
            0.2 * max(0, similarity_now)
        )
        
        # 確保 short_term_rs 在合理範圍內
        short_term_rs = np.nan_to_num(short_term_rs, nan=0.3, posinf=1.0, neginf=0.0)
        short_term_rs = max(0.0, min(1.0, short_term_rs))
        
        client_info_table[cid]["short_term_rs"] = short_term_rs

    # === 4) 計算「步驟三：長期 RS (LongTermScore)」 ===
    #    - 長期表現 (Performance) = accuracy_history 的平均
    #    - 長期可靠度 (Reliability) = 1 - std(accuracy_history)（或其他定義）
    #    - 長期Shapley (Shapley) = shapley_history 的平均
    #    - 最後整合： LongTermScore = w3*Performance + w4*Reliability + w5*Shapley

    w3, w4, w5 = 0.4, 0.3, 0.3  # 權重可自行調整
    for cid in trained_cids:
        info = client_info_table[cid]
        # (a) 取得歷史
        acc_hist = info.get("accuracy_history", [])
        shap_hist = info.get("shapley_history", [])

        # (b) 長期表現 Performance = 平均 accuracy
        if len(acc_hist) > 0:
            longterm_performance = float(np.mean(acc_hist))
        else:
            longterm_performance = 0.0

        # (c) 長期可靠度 Reliability = 1 - std(accuracy)，避免負值則取 max(0, 1 - std)
        if len(acc_hist) >= 2:
            acc_std = float(np.std(acc_hist))
        else:
            acc_std = 0.0
        longterm_reliability = max(0.0, 1.0 - acc_std)

        # (d) 長期 Shapley = shapley_history 平均
        if len(shap_hist) > 0:
            longterm_shapley = float(np.mean(shap_hist))
        else:
            longterm_shapley = 0.0

        # (e) 結合成 LongTermScore
        long_term_score = (
            w3 * longterm_performance +
            w4 * longterm_reliability +
            w5 * longterm_shapley
        )
        
        # 確保 long_term_score 在合理範圍內
        long_term_score = np.nan_to_num(long_term_score, nan=0.3, posinf=1.0, neginf=0.0)
        long_term_score = max(0.0, min(1.0, long_term_score))
        
        client_info_table[cid]["long_term_rs"] = long_term_score

    # (E) **步驟四**：將短期 RS 與長期 RS 整合成最終 Reputation
    #    1) 避免互相壓制，可用單純加權平均：RS_raw = α * ST + (1-α) * LT
    #    2) 做 reliability tuning 或數值裁切
    #    3) 若差異過大，可做懲罰或跳過

    α = 0.5  # 您可自行決定權重
    for cid in trained_cids:
        st_rs = client_info_table[cid]["short_term_rs"]
        lt_rs = client_info_table[cid]["long_term_rs"]
        
        # 確保數值有效
        st_rs = np.nan_to_num(st_rs, nan=0.3, posinf=1.0, neginf=0.0)
        lt_rs = np.nan_to_num(lt_rs, nan=0.3, posinf=1.0, neginf=0.0)

        # (1) 加權
        rs_raw = α * st_rs + (1 - α) * lt_rs

        # (2) 可選：若短期與長期差異過大 => 懲罰或調整
        if lt_rs > 1e-8:  # 避免除以零
            diff_ratio = abs(st_rs - lt_rs) / (lt_rs + 1e-8)
            if diff_ratio > 0.8:
                # 舉例：若差異 > 0.8，將最終值乘以 0.9 懲罰
                rs_raw = rs_raw * 0.9

        # (3) 數值裁切 (避免過大或小)
        rs_final = max(0.0, min(1.0, rs_raw))  # 介於 0~1
        rs_final = np.nan_to_num(rs_final, nan=0.3, posinf=1.0, neginf=0.0)

        # 存到 reputation
        client_info_table[cid]["reputation"] = rs_final

def apply_time_decay(client_info_table: Dict[str, Dict[str, float]], 
                     decay_factor: float = 0.95):
    """在每輪開始前對長期RS施加時間衰減"""
    for cid in client_info_table:
        lt_rs = client_info_table[cid].get("long_term_rs", 0.3)
        # 確保數值有效
        lt_rs = np.nan_to_num(lt_rs, nan=0.3, posinf=1.0, neginf=0.0)
        # 應用衰減，但不讓它低於最小值
        decayed_rs = max(0.1, decay_factor * lt_rs)
        client_info_table[cid]["long_term_rs"] = decayed_rs

def detect_anomalies(
    client_info_table: Dict[str, Dict[str, float]],
    reputation_threshold: float = 0.3,        # 如果低於此值， reputation 異常
    loss_increase_threshold: float = 0.2,       # 如果最新loss比歷史平均高出20%，則認為異常
    similarity_z_threshold: float = -1.0,       # z-score 小於此閾值視為異常
    anomaly_score_threshold: float = 1.5,      # 累計異常得分達到此值則判定為可疑 (提高閾值)
    reputation_weight: float = 1.0,            # 聲譽異常權重
    loss_weight: float = 0.7,                  # 損失異常權重 (降低，因為可能是偶發故障)
    similarity_weight: float = 0.8             # 相似性異常權重
) -> None:
    """
    改進版的異常檢測：
    
    對每個客戶端根據以下指標進行加權檢測：
      1. Reputation：若低於設定閾值則累積異常得分 (權重: reputation_weight)
      2. Loss趨勢：若最新loss相比歷史平均上升超過loss_increase_threshold則累積異常得分 (權重: loss_weight)
      3. 相似性（Similarity）：計算全體客戶端相似性的平均與標準差，對個別客戶端計算z-score，
         若z-score低於 similarity_z_threshold 則累積異常得分 (權重: similarity_weight)
         
    使用加權機制來區分不同類型異常的嚴重性：
    - 聲譽異常 (1.0): 最重要，反映長期表現
    - 相似性異常 (0.8): 重要，可能表示攻擊或數據偏移
    - 損失異常 (0.7): 相對較輕，可能是偶發故障
    
    最後累計異常得分超過 anomaly_score_threshold (預設1.5)，將此客戶端標記為 suspicious。
    """
    import numpy as np

    # 先收集所有客戶端的相似性，計算全局平均與標準差
    similarities = [info.get("similarity", 1.0) for info in client_info_table.values()]
    global_similarity_mean = np.mean(similarities)
    global_similarity_std = np.std(similarities)

    for cid, info in client_info_table.items():
        anomaly_score = 0.0  # 異常得分初始化
        
        # (1) Reputation檢查：低於設定的 reputation_threshold
        rep = info.get("reputation", 1.0)
        if rep < reputation_threshold:
            anomaly_score += reputation_weight

        # (2) Loss 趨勢檢查：最新損失相比於歷史平均是否上升過多
        loss_history = info.get("loss_history", [])
        if len(loss_history) >= 2:
            current_loss = loss_history[-1]
            # 使用除了最新一輪以外的歷史數據計算平均
            historical_avg_loss = np.mean(loss_history[:-1])
            if historical_avg_loss > 0:
                loss_increase_ratio = (current_loss - historical_avg_loss) / historical_avg_loss
                if loss_increase_ratio > loss_increase_threshold:
                    anomaly_score += loss_weight

        # (3) 相似性檢查：計算z-score檢查相似性是否遠低於平均
        sim = info.get("similarity", 1.0)
        if global_similarity_std > 0:
            sim_z = (sim - global_similarity_mean) / global_similarity_std
        else:
            sim_z = 0.0
        if sim_z < similarity_z_threshold:
            anomaly_score += similarity_weight

        # 根據累計異常得分來決定是否標記為可疑
        info["is_suspicious"] = anomaly_score >= anomaly_score_threshold
