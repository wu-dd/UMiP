import hashlib
import logging
import numpy as np
import torch


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
    else:
        sep = "  "

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    res = sep.join([format_val(x) for x in row])
    logging.info(res)


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        if hasattr(self.underlying_dataset, 'root'):  # CUB200: images are loaded from disk, only the shape is needed
            self.data = torch.zeros(5994, 3, 224, 224)
        else:
            self.data = self.underlying_dataset.data[self.keys]
        self.partial_targets = self.underlying_dataset.partial_targets[self.keys]

    def __getitem__(self, key):
        weak_image, strong_image, partial_targets, ord_labels = self.underlying_dataset[self.keys[key]]
        return weak_image, strong_image, partial_targets, ord_labels, key

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def topk_accuracy(network, loader, device, topk=5):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    correct = 0
    total = 0
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            output = network.predict(x)
            _, pred = output.topk(topk, 1, True, True)
            pred = pred.t()
            correct_mat = pred.eq(y.view(1, -1).expand_as(pred))
            correct += correct_mat[:topk].reshape((-1, )).float().sum(0, keepdim=True).item()
            total += len(x)
    network.train()
    return correct / total
