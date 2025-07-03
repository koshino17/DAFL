import torch
import numpy as np
from typing import List, Tuple, Dict
from collections import OrderedDict
from math import log, isnan
from torch.utils.data import DataLoader
import torch.nn as nn

from flwr.client import NumPyClient

from base_FL.train_test import test, UPA_train, TPA_train, train
from base_FL.utils import get_parameters, set_parameters
from base_FL.time import timed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, valloader, train_mode):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.train_mode = train_mode

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        
        if self.train_mode == "UPA":
            UPA_train(self.net, self.trainloader, config)
        elif self.train_mode == "TPA":
            TPA_train(self.net, self.trainloader, config)
        else:
            train(self.net, self.trainloader, config)

        metrics_dict = {
            "partition_id": self.partition_id,  # <--- 額外記錄
        }
        return get_parameters(self.net), len(self.trainloader), metrics_dict

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        
        return float(loss), len(self.valloader), {"client_loss": float(loss), "client_accuracy": float(accuracy), "partition_id": self.partition_id}



