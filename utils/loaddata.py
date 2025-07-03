from typing import Dict, List, Union, Optional
import random
import time
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner

# 全局緩存
_fds_cache: Dict[str, FederatedDataset] = {}
random.seed(1234)

class QProbabilityPartitioner(Partitioner):
    def __init__(self, num_partitions: int, q: float, seed: int = 42):
        super().__init__()
        self._num_partitions = num_partitions
        self.q = q
        self.seed = seed
        self.partitions: Dict[int, List[int]] = {}
        self.raw_dataset: Optional[Dataset] = None  # 保存原始數據集
        random.seed(seed)

    @property
    def num_partitions(self) -> int:
        return self._num_partitions

    def load_partition(self, partition_id: int) -> Dataset:
        if self.raw_dataset is None or not self.partitions:
            raise RuntimeError("分區未初始化，請先調用 split()")
        return self.raw_dataset.select(self.partitions[partition_id])  # 使用 select 方法

    def split(self, dataset: Dataset) -> Dict[int, List[int]]:
        self.raw_dataset = dataset  # 保存原始數據集
        
        # 檢測標籤欄位
        label_field = "label"
        if label_field not in dataset.column_names and "fine_label" in dataset.column_names:
            label_field = "fine_label"
            
        num_classes = len(set(dataset[label_field]))
        class_indices = self._split_by_class(dataset, num_classes, label_field)
        self.partitions = self._assign_data(class_indices, num_classes)
        return self.partitions

    def _split_by_class(self, dataset: Dataset, num_classes: int, label_field: str) -> Dict[int, List[int]]:
        return {i: [idx for idx, ex in enumerate(dataset) if ex[label_field] == i] 
                for i in range(num_classes)}

    def _assign_data(self, class_indices: Dict[int, List[int]], num_classes: int) -> Dict[int, List[int]]:
        partitions = {i: [] for i in range(self.num_partitions)}
        group_per_class, remaining = divmod(self.num_partitions, num_classes)
        
        # 動態分配組數
        group_assignments = []
        current_group = 0
        for class_id in range(num_classes):
            groups = group_per_class + (1 if class_id < remaining else 0)
            group_assignments.append((
                class_id, 
                current_group, 
                current_group + groups
            ))
            current_group += groups

        # 分配樣本
        for class_id, start, end in group_assignments:
            for idx in class_indices.get(class_id, []):
                if random.random() < self.q:
                    target = random.randint(start, end-1)
                else:
                    others = [i for i in range(self.num_partitions) if not (start <= i < end)]
                    target = random.choice(others)
                partitions[target].append(idx)
        return partitions

# def get_transforms(dataset_name: str) -> transforms.Compose:
#     """根據 dataset_name 回傳 torchvision transforms"""
#     name = dataset_name.lower()
#     if name == "cifar10":
#         mean = (0.4914, 0.4822, 0.4465)
#         std = (0.2023, 0.1994, 0.2010)
#         base = [
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomCrop(32, padding=4),
#         ]
#     elif name == "cifar100":
#         mean = (0.5071, 0.4867, 0.4408)
#         std = (0.2675, 0.2565, 0.2761)
#         base = [
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomCrop(32, padding=4),
#         ]
#     elif name == "mnist":
#         mean = (0.1307, 0.1307, 0.1307)
#         std = (0.3081, 0.3081, 0.3081)
#         base = [
#             transforms.Resize(32),
#             transforms.Grayscale(num_output_channels=3),
#             transforms.RandomCrop(32, padding=4),
#         ]
#     elif name == "svhn":
#         mean = (0.4377, 0.4438, 0.4728)
#         std = (0.1980, 0.2010, 0.1970)
#         base = [
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomCrop(32, padding=4),
#         ]
#     else:
#         raise ValueError(f"Unsupported dataset: {dataset_name}")
    
#     return transforms.Compose([*base, transforms.ToTensor(), transforms.Normalize(mean, std)])
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
    elif name == "fashion_mnist":
        mean = (0.2860, 0.2860, 0.2860)
        std = (0.3530, 0.3530, 0.3530)
        base = [
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomCrop(32, padding=4),
        ]
    elif name == "svhn":
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        base = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return transforms.Compose([*base, transforms.ToTensor(), transforms.Normalize(mean, std)])

def apply_transforms(batch: Dict, tfm: transforms.Compose) -> Dict:
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
    dataset_name: str = "mnist",
    q: float = 1.0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    支援 cifar10, cifar100, mnist, svhn 等資料集的聯邦學習資料載入器
    
    Args:
        partition_id: 要載入的分區ID
        batch_size: 批次大小
        num_partitions: 分區數量
        dataset_name: 資料集名稱，支援 "mnist", "cifar10", "cifar100", "svhn"
        q: Q概率值，1.0表示完全依照類別分組，0.0表示完全隨機分配
    """
    if isinstance(partition_id, str):
        partition_id = int(partition_id)
        
    cache_key = f"{dataset_name}-{num_partitions}-{q}"
    
    if cache_key not in _fds_cache:
        print(f"[{time.ctime()}] 初始化聯邦數據集: {dataset_name}, 分區數={num_partitions}, q={q}")
        
        # 針對SVHN設定subset
        subset = "cropped_digits" if dataset_name.lower() == "svhn" else None
        
        # 加載原始數據集並分區
        raw_train = load_dataset(dataset_name, subset, split="train")
        partitioner = QProbabilityPartitioner(num_partitions, q, seed=42)
        partitioner.split(raw_train)  # 這裡會自動保存 raw_dataset
        
        # 創建聯邦數據集
        _fds_cache[cache_key] = FederatedDataset(
            dataset=dataset_name,
            subset=subset,
            partitioners={"train": partitioner}
        )
    
    fds = _fds_cache[cache_key]
    
    partition = fds.load_partition(partition_id).train_test_split(test_size=0.2, seed=42)
    tfm = get_transforms(dataset_name)
    
    # 應用數據轉換
    partition = partition.with_transform(lambda b: apply_transforms(b, tfm))
    test_data = fds.load_split("test").with_transform(lambda b: apply_transforms(b, tfm))
    
    # 創建 DataLoader
    train_loader = DataLoader(partition["train"], batch_size=batch_size, 
                              shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(partition["test"], batch_size=batch_size, 
                            shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, 
                             shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader, test_loader

# 測試代碼
from collections import Counter
def main(
    num_partitions: int = 10,
    dataset_name: str = "mnist",
    q: float = 1.0,
    batch_size: int = 64,
):
    cache_key = f"{dataset_name}-{num_partitions}-{q}"
    for pid in range(num_partitions):
        # 取得第 pid 個分區的 DataLoader
        train_dl, val_dl, test_dl = get_cached_datasets(
            partition_id=pid,
            batch_size=batch_size,
            num_partitions=num_partitions,
            dataset_name=dataset_name,
            q=q
        )
        # 計算樣本與批次數量
        num_train = len(train_dl.dataset)
        num_val = len(val_dl.dataset)
        num_test = len(test_dl.dataset)
        n_tr = len(train_dl)
        n_val = len(val_dl)
        n_test = len(test_dl)

        # 取得原始分區並計算標籤分佈
        raw_part = _fds_cache[cache_key].load_partition(pid)
        labels = raw_part["label"]
        dist = Counter(labels)
        
        # 新增：取得原始測試集並計算標籤分佈（test）
        raw_test = _fds_cache[cache_key].load_split("test")
        dist_test = Counter(raw_test["label"])

        # print(f"測試集標籤分布: {dist_test}")

        # 列印結果
        print(f"=== Partition {pid} / {num_partitions - 1} ===")
        print(f"標籤分佈: {dist}")
        # print(f"訓練集大小: {num_train}")
        # print(f"驗證集大小: {num_val}")
        # print(f"測試集大小: {num_test}")
        # print(f"訓練集批次數: {n_tr}")
        # print(f"驗證集批次數: {n_val}")
        # print(f"測試集批次數: {n_test}")
        # print(f"train + val + test = {num_train + num_val + num_test}\n")


if __name__ == "__main__":
    # 若需調整參數，可修改以下呼叫
    main(num_partitions=100, dataset_name="fashion_mnist", q=0.9, batch_size=64)

