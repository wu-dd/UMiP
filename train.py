import argparse
import collections
import os
import random
import sys
import logging
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from data import datasets
from core import hparams_registry, algorithms
from lib import misc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UMiP: partial-label learning with uncertainty-aware mixup')
    # dataset
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default="CIFAR10", choices=["SVHN", "CIFAR10", "CIFAR100", "CUB200", "FMNIST", "KMNIST"])

    # model
    parser.add_argument('--model', type=str, default='ResNet32', choices=["LeNet", "ResNet32", "ResNet34", "ResNet110"])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=400)

    # partial-label generation
    parser.add_argument('--pl_type', type=str, default='proden_uniform', choices=['proden_uniform', 'uniform', 'class_dependent', 'instance_dependent', 'aggregate', 'vaguest', 'sup'],
                        help='how candidate label sets are generated')
    parser.add_argument('--pl_degree', type=float, default=0, help='flip rate / ambiguity degree of the candidate label generation')

    # base PLL algorithm and whether UMiP is plugged in
    parser.add_argument('--algorithm', type=str, default="PRODEN", choices=["PRODEN", "LWS", "CRDPLL", "PiCO", "DIRK", "ABLE"])
    parser.add_argument('--mode', type=str, default="orig", choices=["orig", "UMiP"],
                        help='"orig": run the base algorithm as is; "UMiP": base algorithm + UMiP')

    # seeds
    parser.add_argument('--trial_seed', type=int, default=0, help='Seed for the train/holdout split and random hparams.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--holdout_fraction', type=float, default=0.0)

    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.StreamHandler()])

    logging.info("Environment:")
    logging.info("\tPython: {}".format(sys.version.split(" ")[0]))
    logging.info("\tPyTorch: {}".format(torch.__version__))
    logging.info("\tTorchvision: {}".format(torchvision.__version__))
    logging.info("\tCUDA: {}".format(torch.version.cuda))
    logging.info("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    logging.info("\tNumPy: {}".format(np.__version__))
    logging.info("\tPIL: {}".format(PIL.__version__))

    logging.info('Args:')
    for k, v in sorted(vars(args).items()):
        logging.info('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, misc.seed_hash(args.hparams_seed, args.trial_seed))

    hparams['model'] = args.model
    hparams['lr'] = args.lr
    hparams['weight_decay'] = args.weight_decay
    hparams['epochs'] = args.epochs
    hparams['batch_size'] = args.batch_size
    hparams['mode'] = args.mode

    logging.info('HParams:')
    for k, v in sorted(hparams.items()):
        logging.info('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset in datasets.IMAGE_DATASETS:
        dataset, test_dataset = datasets.get_dataset(args)
    else:
        raise NotImplementedError

    val_dataset, train_dataset = misc.split_dataset(dataset, int(len(dataset) * args.holdout_fraction), misc.seed_hash(args.trial_seed))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(train_dataset.data.shape, train_dataset.partial_targets, hparams)
    algorithm.to(device)

    checkpoint_vals = collections.defaultdict(lambda: [])
    last_results_keys = None
    for epoch in range(args.epochs):
        algorithm.train()
        for minibatches_device in train_loader:
            minibatches_device = [item.to(device) for item in minibatches_device]
            step_vals = algorithm.update(minibatches_device, epoch)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        results = {'epoch': epoch}
        for key, val in checkpoint_vals.items():
            results[key] = np.mean(val)

        acc = misc.topk_accuracy(algorithm, test_loader, device, topk=1)
        results['test_acc'] = acc
        results_keys = sorted(results.keys())
        if results_keys != last_results_keys:
            misc.print_row(results_keys, colwidth=12)
            last_results_keys = results_keys
        misc.print_row([results[key] for key in results_keys], colwidth=12)
