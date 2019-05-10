import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cross_validation_with_val_set(dataset,
                                  model,
                                  folds,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  logger=None):

    val_losses, accs, durations = [], [], []
    for fold, (train_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds))):

        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_loss = train(model, optimizer, train_loader, device)
            val_losses.append(eval_loss(model, val_loader, device))
            accs.append(eval_acc(model, val_loader, device))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_losses[-1],
                'val_acc': accs[-1],
            }

            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(folds, dtype=torch.long), argmin]

    print('Val Loss: {:.4f}, Val Accuracy: {:.4f} ± {:.4f}, Duration: {:.4f}'.
          format(loss.mean().item(),
                 acc.mean().item(),
                 acc.std().item(),
                 duration.mean().item()))

    return loss.mean().item(), acc.mean().item(), acc.std().item()


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    val_indices, train_indices = [], []

    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        val_indices.append(torch.from_numpy(idx))

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, device):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader, device):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)
