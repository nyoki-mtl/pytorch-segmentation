import torch.optim as optim
from .scheduler import CosineWithRestarts


def create_optimizer(model, mode='adam', base_lr=1e-3, t_max=10):
    if mode == 'adam':
        optimizer = optim.Adam(model.parameters(), base_lr)
    elif mode == 'sgd':
        optimizer = optim.SGD(model.parameters(), base_lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise NotImplementedError(mode)

    scheduler = CosineWithRestarts(optimizer, t_max)

    return optimizer, scheduler
