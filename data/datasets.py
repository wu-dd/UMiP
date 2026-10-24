import os
import torch
from torchvision import transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset
import pickle
import numpy as np
from scipy.special import comb
from data.randaugment import RandomAugment
from data.cutout import Cutout
from torchvision import models
import torch.nn as nn
import logging
import torch.nn.functional as F
from data.partial_models_pool.wide_resnet import WideResNet
from data.partial_models_pool.mlp import mlp_partialize
from torchvision.datasets.folder import pil_loader
from data.dataset_cub import Cub2011

IMAGE_DATASETS = [
    "CIFAR10",
    "CIFAR100",
    "CUB200",
    "FMNIST",
    "KMNIST",
    "SVHN",
]


def prepare_cv_datasets(dataname, batch_size):
    if dataname == 'KMNIST':
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ])
        test_dataset = dsets.KMNIST(root='./data/KMNIST', train=False, transform=test_transform)
        ordinary_train_dataset = dsets.KMNIST(root='./data/KMNIST', train=True, transform=transforms.ToTensor(),download=True)
        ori_data, ori_labels = ordinary_train_dataset.data, ordinary_train_dataset.targets.long()
        ori_data = ori_data.view((ori_data.shape[0], -1)).float()
        vanilla_data = ordinary_train_dataset.data
    elif dataname == 'FMNIST':
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081)),
        ])
        test_dataset = dsets.FashionMNIST(root='./data/FMNIST', train=False, transform=test_transform)
        ordinary_train_dataset = dsets.FashionMNIST(root='./data/FMNIST', train=True, transform=transforms.ToTensor(), download=True)
        ori_data, ori_labels = ordinary_train_dataset.data, ordinary_train_dataset.targets.long()
        ori_data = ori_data.view((ori_data.shape[0], -1)).float()
        vanilla_data = ordinary_train_dataset.data
    elif dataname == 'SVHN':
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        test_dataset = dsets.SVHN(root='./data/SVHN', split="test", transform=test_transform,download=True)
        ordinary_train_dataset = dsets.SVHN(root='./data/SVHN', split="train", transform=transforms.ToTensor(),download=True)
        ori_data, ori_labels = torch.Tensor(ordinary_train_dataset.data), torch.Tensor(ordinary_train_dataset.labels).long()
        ori_data=ori_data
        vanilla_data = ordinary_train_dataset.data.transpose(0, 2, 3, 1)

    elif dataname == 'CIFAR10':
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        test_dataset = dsets.CIFAR10(root='./data/CIFAR10', train=False, transform=test_transform)
        ordinary_train_dataset = dsets.CIFAR10(root='./data/CIFAR10', train=True, transform=transforms.ToTensor(), download=True)
        ori_data, ori_labels = torch.Tensor(ordinary_train_dataset.data), torch.Tensor(ordinary_train_dataset.targets).long()
        ori_data = ori_data.permute(0, 3, 1, 2)
        vanilla_data = ordinary_train_dataset.data
    elif dataname == 'CIFAR100':
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        test_dataset = dsets.CIFAR100(root='./data/CIFAR100', train=False, transform=test_transform)
        ordinary_train_dataset = dsets.CIFAR100(root='./data/CIFAR100', train=True, transform=transforms.ToTensor(), download=True)
        ori_data, ori_labels = torch.Tensor(ordinary_train_dataset.data), torch.Tensor(ordinary_train_dataset.targets).long()
        ori_data = ori_data.permute(0, 3, 1, 2)
        vanilla_data = ordinary_train_dataset.data

    elif dataname == 'CUB200':
        test_transform = transforms.Compose([
            transforms.Resize(int(224 / 0.875)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        test_dataset = Cub2011('./data/CUB200', train=False, transform=test_transform, loader=pil_loader)
        ordinary_train_dataset = Cub2011('./data/CUB200', train=True, transform=test_transform, loader=pil_loader)
        original_full_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=len(ordinary_train_dataset),shuffle=False, num_workers=20)
        ori_data, ori_labels = next(iter(original_full_loader))
        ori_labels = ori_labels.long()
        vanilla_data = [ordinary_train_dataset.root,ordinary_train_dataset.base_folder,ordinary_train_dataset.data]

    if dataname == 'CIFAR100':
        num_classes=100
    elif dataname == 'CUB200':
        num_classes=200
    else:
        num_classes = 10
    num_samples=len(ordinary_train_dataset)
    return ori_data, vanilla_data, ori_labels, test_dataset, num_samples, num_classes


def generate_proden_uniform_cv_candidate_labels(train_labels,partial_degree=0.5):
    c = torch.max(train_labels) - torch.min(train_labels) + 1
    n = train_labels.shape[0]

    partialY = F.one_hot(train_labels, num_classes=c)
    for i in range(n):
        row = partialY[i, :]
        row[np.where(np.random.binomial(1, partial_degree, c) == 1)] = 1
        while torch.sum(row) == 1:
            row[np.random.randint(0, c)] = 1
        partialY[i] = row

    logging.info("Finish Generating Candidate Label Sets!\n")
    return partialY


def generate_uniform_cv_candidate_labels(train_labels):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = torch.max(train_labels) - torch.min(train_labels) + 1
    n = train_labels.shape[0]
    cardinality = (2 ** K - 2).float()
    number = torch.tensor([comb(K, i + 1) for i in range(
        K - 1)]).float()  # 1 to K-1 because cannot be empty or full label set, convert list to tensor
    frequency_dis = number / cardinality
    prob_dis = torch.zeros(K - 1)  # tensor of K-1
    for i in range(K - 1):
        if i == 0:
            prob_dis[i] = frequency_dis[i]
        else:
            prob_dis[i] = frequency_dis[i] + prob_dis[i - 1]

    random_n = torch.from_numpy(np.random.uniform(0, 1, n)).float()  # tensor: n
    mask_n = torch.ones(n)  # n is the number of train_data
    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0

    temp_num_partial_train_labels = 0  # save temp number of partial train_labels

    for j in range(n):  # for each instance
        for jj in range(K - 1):  # 0 to K-2
            if random_n[j] <= prob_dis[jj] and mask_n[j] == 1:
                temp_num_partial_train_labels = jj + 1  # decide the number of partial train_labels
                mask_n[j] = 0

        temp_num_fp_train_labels = temp_num_partial_train_labels - 1
        candidates = torch.from_numpy(np.random.permutation(K.item())).long()  # because K is tensor type
        candidates = candidates[candidates != train_labels[j]]
        temp_fp_train_labels = candidates[:temp_num_fp_train_labels]

        partialY[j, temp_fp_train_labels] = 1.0  # fulfill the partial label matrix
    logging.info("Finish Generating Candidate Label Sets!\n")
    return partialY


def generate_class_dependent_cv_candidate_labels(train_labels,partial_degree=0.5):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1
    pass
    K = torch.max(train_labels) - torch.min(train_labels) + 1
    assert K == 10
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0

    q_adj = partial_degree
    transition_matrix = [
        [1, q_adj, 0, 0, 0, 0, 0, 0, q_adj, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, q_adj, q_adj],
        [0, 0, 1, q_adj, q_adj, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, q_adj, q_adj, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, q_adj, 0, q_adj, 0, 0],
        [0, 0, 0, q_adj, q_adj, 1, 0, 0, 0, 0],
        [0, 0, q_adj, q_adj, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, q_adj, q_adj, 0, 1, 0, 0],
        [0, q_adj, 0, 0, 0, 0, 0, 0, 1, q_adj],
        [0, q_adj, 0, 0, 0, 0, 0, 0, q_adj, 1],
    ]

    transition_matrix = np.array(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K):  # for each class
            if jj == train_labels[j]:  # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0

    logging.info("Finish Generating Candidate Label Sets!\n")
    return partialY


def generate_instance_dependent_cv_cancidate_labels(dataname, ori_data,ori_labels,partial_degree):
    if dataname == 'CIFAR10':
        weight_path = './data/partial_models_pool/partial_weights/cifar10.pt'
        model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
        rate = partial_degree # 0.4
    elif dataname == 'CIFAR100':
        weight_path = './data/partial_models_pool/partial_weights/cifar100.pt'
        model = models.wide_resnet50_2()
        model.fc = nn.Linear(model.fc.in_features, max(ori_labels) + 1)
        model = torch.nn.DataParallel(model)
        rate = partial_degree # 0.04
    elif dataname == 'KMNIST':
        weight_path = './data/partial_models_pool/partial_weights/kmnist.pt'
        model = mlp_partialize(n_inputs=28 * 28, n_outputs=max(ori_labels) + 1)
        rate = partial_degree  # 0.4
    elif dataname == 'FMNIST':
        weight_path = './data/partial_models_pool/partial_weights/fmnist.pt'
        model = mlp_partialize(n_inputs=28*28, n_outputs=max(ori_labels) + 1)
        rate = partial_degree #0.4
    elif dataname == 'CUB200':
        weight_path = './data/partial_models_pool/partial_weights/cub200.pt'
        model = models.wide_resnet50_2()
        model.fc = nn.Linear(model.fc.in_features, max(ori_labels) + 1)
        model = torch.nn.DataParallel(model)
        rate = partial_degree
    elif dataname == 'SVHN':
        weight_path = './data/partial_models_pool/partial_weights/svhn.pt'
        model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
        rate = partial_degree  # 0.4

    with torch.no_grad():
        model.load_state_dict(torch.load(weight_path,map_location='cuda:0'))
        K=int(torch.max(ori_labels)) + 1
        n = ori_labels.shape[0]
        model = model.cuda()
        train_Y = torch.nn.functional.one_hot(ori_labels, num_classes=K)
        partialY_list = []
        outputs_list = []
        batch_size = 2000
        import math
        step = math.ceil(n / batch_size)

        for i in range(0, step):
            b_end = min((i + 1) * batch_size, n)
            train_X_part = ori_data[i * batch_size: b_end].cuda()
            outputs = model(train_X_part)
            outputs_list.append(outputs)
            train_p_Y = train_Y[i * batch_size: b_end].clone().detach()
            partial_rate_array = F.softmax(outputs, dim=1).clone().detach()
            partial_rate_array[torch.where(train_p_Y == 1)] = 0
            partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
            partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
            partial_rate_array[partial_rate_array > 1.0] = 1.0
            m = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array)
            z = m.sample()
            train_p_Y[torch.where(z == 1)] = 1.0
            partialY_list.append(train_p_Y)
        partialY = torch.cat(partialY_list, dim=0).float()
        outputs_list=torch.cat(outputs_list, dim=0)
        logging.info("Finish Generating Candidate Label Sets!\n")
        return partialY.cpu()


def generate_real_id_cv_cancidate_labels(partial_type, orig_data,labels,K):
    dataset_path = './data/plcifar10/plcifar10.pkl'
    partial_label_all = pickle.load(open(dataset_path, 'rb'))
    partial_targets = np.zeros((len(orig_data), K))
    if partial_type == 'aggregate':
        for key, value in partial_label_all.items():
            for candidate_label_set in value:
                for label in candidate_label_set:
                    partial_targets[key, label] = 1
    elif partial_type == 'vaguest':
        for key, value in partial_label_all.items():
            vaguest_candidate_label_set = []
            largest_num = 0
            for candidate_label_set in value:
                if len(candidate_label_set) > largest_num:
                    vaguest_candidate_label_set = candidate_label_set
                    largest_num = len(candidate_label_set)
            for label in vaguest_candidate_label_set:
                partial_targets[key, label] = 1
    partialY = partial_targets
    partialY = torch.from_numpy(partialY)
    for i in range(K):
        idx = torch.where(labels == i)[0]
        partialY[idx, i] = 1
    logging.info("Finish Generating Candidate Label Sets!\n")
    return partialY


def generate_sup_cv_cancidate_labels(labels,K):
    partialY = torch.nn.functional.one_hot(labels, num_classes=K).float()
    return partialY


class gen_index_dataset(Dataset):
    def __init__(self, dataname, images, given_label_matrix, true_labels):
        self.dataname=dataname
        if dataname == 'CUB200':
            self.root, self.base_folder, self.data=images
            self.loader = pil_loader
        else:
            self.data = images
        self.partial_targets = given_label_matrix
        self.ord_labels = true_labels

        if dataname=="FMNIST":
            self.train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081]),
            ])
            self.Aug_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.Normalize(mean=[0.1307], std=[0.3081]),
            ])
        elif dataname=="KMNIST":
            self.train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
            self.Aug_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        elif dataname=="CIFAR10":
            self.train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            self.Aug_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                RandomAugment(3, None),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        elif dataname=="CIFAR100":
            self.train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

            self.Aug_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                RandomAugment(3, None),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        elif dataname=="SVHN":
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.Aug_transform = transforms.Compose([
                transforms.ToPILImage(),
                RandomAugment(3, None),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif dataname=="CUB200":
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])
            self.Aug_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                RandomAugment(3, 5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
            ])

    def __len__(self):
        return len(self.ord_labels)

    def __getitem__(self, index):
        if self.dataname=='CUB200':
            sample = self.data.iloc[index]
            path = os.path.join(self.root, self.base_folder, sample.filepath)
            img = self.loader(path)
            each_image = self.train_transform(img)
            each_aug_image = self.Aug_transform(img)
            each_label = self.partial_targets[index]
            each_true_label = self.ord_labels[index]
        else:
            each_image = self.train_transform(self.data[index])
            each_aug_image = self.Aug_transform(self.data[index])
            each_label = self.partial_targets[index]
            each_true_label = self.ord_labels[index]

        return each_image, each_aug_image, each_label, each_true_label


def prepare_train_loaders_for_candidate_labels(dataname, ori_data, vanilla_data, ori_labels, partial_type,partial_degree):

    K=max(ori_labels)+1
    if partial_type == 'uniform':
        partialY = generate_uniform_cv_candidate_labels(ori_labels)
    elif partial_type == 'proden_uniform':
        partialY = generate_proden_uniform_cv_candidate_labels(ori_labels, partial_degree)
    elif partial_type == 'class_dependent':
        partialY = generate_class_dependent_cv_candidate_labels(ori_labels, partial_degree)
    elif partial_type == 'instance_dependent':
        partialY = generate_instance_dependent_cv_cancidate_labels(dataname, ori_data,ori_labels,partial_degree)
    elif partial_type in ['aggregate','vaguest']:
        partialY = generate_real_id_cv_cancidate_labels(partial_type, ori_data, ori_labels,K)
    elif partial_type == 'sup':
        partialY = generate_sup_cv_cancidate_labels(ori_labels,K)

    partial_matrix_dataset = gen_index_dataset(dataname,vanilla_data, partialY.float(), ori_labels.float())
    return partial_matrix_dataset


def get_dataset(args):
    ori_data, vanilla_data, ori_labels, test_dataset, num_samples, K = prepare_cv_datasets(dataname=args.dataset, batch_size=args.batch_size)
    train_dataset = prepare_train_loaders_for_candidate_labels(dataname=args.dataset,ori_data=ori_data, vanilla_data=vanilla_data, ori_labels=ori_labels, partial_type=args.pl_type,partial_degree=args.pl_degree)
    return train_dataset, test_dataset
