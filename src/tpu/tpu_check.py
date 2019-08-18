import os

os.environ['XRT_TPU_CONFIG'] = 'tpu_worker;0;10.0.101.2:8470'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch_xla
import torch_xla_py.data_parallel as dp
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm


num_cores = None

devices = (xm.get_xla_supported_devices(max_devices=num_cores) if num_cores != 0 else [])
print(devices)
