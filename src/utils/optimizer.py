import torch.optim as optim
from .scheduler import CosineWithRestarts


def create_optimizer(params, mode='adam', base_lr=1e-3, t_max=10):
    if mode == 'adam':
        optimizer = optim.Adam(params, base_lr)
    elif mode == 'sgd':
        optimizer = optim.SGD(params, base_lr, momentum=0.9, weight_decay=4e-5)
    else:
        raise NotImplementedError(mode)

    scheduler = CosineWithRestarts(optimizer, t_max)

    return optimizer, scheduler
