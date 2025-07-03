import time
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from flwr_datasets import FederatedDataset


fds = None  # Cache FederatedDataset
def load_datasets(partition_id: int, BATCH_SIZE: int, NUM_CLIENTS: int):
    print(f"Loading dataset for partition {partition_id} at {time.time()}")
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # Cutout
    ])

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    testset = fds.load_split("test").with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)
    return trainloader, valloader, testloader

def get_cached_datasets(partition_id: int, BATCH_SIZE: int = 64, num_partitions: int =10):
    global fds
    if fds is None:
        print(f"Loading dataset for partition {partition_id} at {time.time()}")
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # Cutout
    ])

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    testset = fds.load_split("test").with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)
    # testloader = DataLoader(testset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)
    return trainloader, testloader