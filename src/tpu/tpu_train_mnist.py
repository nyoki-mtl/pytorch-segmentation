import os
os.environ['XRT_TPU_CONFIG'] = 'tpu_worker;0;10.0.101.2:8470'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch_xla
import torch_xla_py.data_parallel as dp
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm

datadir = '/tmp/mnist-data'
num_cores = None
batch_size = 128
num_epochs = 18
num_workers = 4
log_steps = 20
base_lr = 0.01
momentum = 0.5
target_accuracy = 98.0

metrics_debug = True

torch.manual_seed(1)
torch.set_default_tensor_type('torch.FloatTensor')


class MNIST(nn.Module):

    def __init__(self):
        super(MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train_loop_fn(model, loader, device, context):
    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    tracker = xm.RateTracker()

    # model.train()
    for x, (data, target) in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        xm.optimizer_step(optimizer)
        tracker.add(batch_size)
        if x % log_steps == 0:
            print(f'[{device}]({x}) Loss={loss.item():.5f} Rate={tracker.rate():.2f}')


def test_loop_fn(model, loader, device, context):
    total_samples = 0
    correct = 0

    # model.eval()
    for x, (data, target) in loader:
        output = model(data)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size()[0]

    print(f'[{device}] Accuracy={100.0 * correct / total_samples:.2f}%')
    return correct / total_samples


transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(datadir, train=True, download=True, transform=transformer)
test_dataset = datasets.MNIST(datadir, train=False, transform=transformer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

devices = xm.get_xla_supported_devices(max_devices=num_cores)
# Scale learning rate to num cores
lr = base_lr * len(devices)
# Pass [] as device_ids to run using the PyTorch/CPU engine.
model_parallel = dp.DataParallel(MNIST, device_ids=devices)

accuracy = 0.0
for epoch in range(1, num_epochs + 1):
    model_parallel(train_loop_fn, train_loader)
    accuracies = model_parallel(test_loop_fn, test_loader)
    accuracy = sum(accuracies) / len(accuracies)
    if metrics_debug:
        print(torch_xla._XLAC._xla_metrics_report())

print(accuracy * 100.0)
