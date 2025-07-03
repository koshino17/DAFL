import time
from typing import Union
import random

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets.partitioner import ShardPartitioner
from flwr_datasets.partitioner import DirichletPartitioner
# 全域快取字典
_fds_cache: dict[str, FederatedDataset] = {}
# ALPHA = 0.1000
random.seed(1234)

def get_transforms(dataset_name: str) -> transforms.Compose:
    """根據 dataset_name 回傳 torchvision transforms"""
    name = dataset_name.lower()
    if name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        base = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
        ]
    elif name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        base = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
        ]
    elif name == "mnist":
        mean = (0.1307, 0.1307, 0.1307)
        std = (0.3081, 0.3081, 0.3081)
        base = [
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomCrop(32, padding=4),
        ]
    elif name == "svhn":
        # HF SVHN 預設兩個 config：cropped_digits / full_numbers
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        base = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return transforms.Compose([*base, transforms.ToTensor(), transforms.Normalize(mean, std)])

def apply_transforms(batch: dict, tfm: transforms.Compose) -> dict:
    """
    找出 batch 中的影像欄位(image 或 img)，轉成 Tensor，統一放到 batch['img']，移除原欄位。
    """
    # 偵測原有欄位
    if "img" in batch:
        src = "img"
    elif "image" in batch:
        src = "image"
    else:
        raise KeyError(f"No image field in batch: {batch.keys()}")
    # 轉換
    batch["img"] = [tfm(x) for x in batch[src]]
    # 刪除舊欄位
    if src != "img":
        del batch[src]
        
    # 標籤部分：如果是 CIFAR-100，就把 fine_label 放到 label
    if "fine_label" in batch:
        batch["label"] = [int(l) for l in batch.pop("fine_label")]
    return batch

def get_cached_datasets(
    partition_id: Union[int, str],
    batch_size: int = 64,
    num_partitions: int = 10,
    dataset_name: str = "mnist", # 支援 cifar10, cifar100, mnist, svhn
    ALPHA = 0.1000
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    只在第一次建立 FederatedDataset，之後重複使用同一份快取，用於多輪 FL 模擬。
    支援 cifar10, cifar100, mnist, svhn。
    """
    if isinstance(partition_id, str):
        partition_id = int(partition_id)
    cache_key = f"{dataset_name}-{num_partitions}"
    if cache_key not in _fds_cache:
        print(f"[{time.time():.0f}] Caching {dataset_name} n={num_partitions}")
        subset = "cropped_digits" if dataset_name.lower() == "svhn" else None
        # partitioner = IidPartitioner(num_partitions=num_partitions)
        # 假设 dataset_name 有 M 个类别，你也可以动态获取
        num_classes = {"mnist":10, "cifar10":10, "cifar100":100, "svhn":10}[dataset_name]
        # 假设 num_partitions=10，每个 client 都有标签 'label'
        # alpha=0.1 → 强非IID；alpha=1000 → 近似IID
        partitioner = DirichletPartitioner(
            num_partitions=10,
            partition_by="label",
            alpha=ALPHA,
            min_partition_size=5,      # 每个 client 最少保留多少样本
            self_balancing=False,
            seed=42,
        )
        _fds_cache[cache_key] = FederatedDataset(
            dataset=dataset_name,
            subset=subset,
            partitioners={"train": partitioner},
        )
    fds = _fds_cache[cache_key]

    # # 看資料分布
    # from collections import Counter
    # import pandas as pd
    # for pid in range(num_partitions):
    #     part_i = fds.load_partition(pid)
    #     labels_i = [ex["label"] for ex in part_i]
    #     print(f"Partition {pid}: {Counter(labels_i)}")
    # # 看資料分布並存成 CSV
    # num_classes = {"mnist":10, "cifar10":10, "cifar100":100, "svhn":10}[dataset_name.lower()]
    # # 準備一個 dict: { client_id: Counter(...) }
    # distribution = {}
    # for pid in range(num_partitions):
    #     part_i = fds.load_partition(pid)
    #     labels_i = [ex["label"] for ex in part_i]
    #     distribution[pid] = Counter(labels_i)

    # # 轉成 DataFrame：列為 client_id，欄為 class，缺值補 0
    # df = pd.DataFrame.from_dict(
    #     distribution,
    #     orient="index",                      # keys 作為 index
    #     columns=list(range(num_classes))     # 0,1,2,…,num_classes-1
    # ).fillna(0).astype(int)

    # df.index.name = "client_id"
    # # 把 CSV 寫到 working directory
    # csv_path = "partition_distribution.csv"
    # df.to_csv(csv_path)
    # print(f"已儲存 partition distribution 到 {csv_path}")
    
    part = fds.load_partition(partition_id)
    part = part.train_test_split(test_size=0.2, seed=42)

    tfm = get_transforms(dataset_name)
    part = part.with_transform(lambda b: apply_transforms(b, tfm))
    test_all = fds.load_split("test") \
                .with_transform(lambda b: apply_transforms(b, tfm))

    train_dl = DataLoader(
        part["train"], batch_size=batch_size,
        shuffle=True, num_workers=8, pin_memory=True
    )
    val_dl = DataLoader(
        part["test"], batch_size=batch_size,
        shuffle=False, num_workers=8, pin_memory=True
    )
    test_dl = DataLoader(
        test_all, batch_size=batch_size,
        shuffle=False, num_workers=8, pin_memory=True
    )
    return train_dl, val_dl, test_dl
