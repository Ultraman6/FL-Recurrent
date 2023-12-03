import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from .without_reload import FashionMNIST_truncated, FashionMNIST_truncated_WO_reload


# generate the non-IID distribution for all methods
def read_data_distribution(filename="./data_preprocessing/non-iid-distribution/FashionMNIST/distribution.txt",):
    distribution = {}
    with open(filename, "r") as data:
        for x in data.readlines():
            if "{" != x[0] and "}" != x[0]:
                tmp = x.split(":")
                if "{" == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(",", ""))
    return distribution


def read_net_dataidx_map(filename="./data_preprocessing/non-iid-distribution/FashionMNIST/net_dataidx_map.txt",):
    net_dataidx_map = {}
    with open(filename, "r") as data:
        for x in data.readlines():
            if "{" != x[0] and "}" != x[0] and "]" != x[0]:
                tmp = x.split(":")
                if "[" == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(",")
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug("Data statistics: %s" % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_fashionMNIST():
    CIFAR_MEAN = [0.49139968,]
    CIFAR_STD = [0.24703233,]

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            # transforms.Grayscale(num_output_channels=3),  # 3为三通道，1为单通道
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            Cutout(16)
        ]
    )

    valid_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

    return train_transform, valid_transform


def load_fashionMNIST_data(datadir, process_id, synthetic_data_url, private_local_data, resize=32, augmentation=True, data_efficient_load=False):
    train_transform, test_transform = _data_transforms_fashionMNIST()

    is_download = True

    if data_efficient_load:
        fashionMNIST_train_ds = FashionMNIST(datadir, train=True, download=True, transform=train_transform)
        fashionMNIST_test_ds = FashionMNIST(datadir, train=False, download=True, transform=test_transform)
    else:
        fashionMNIST_train_ds = FashionMNIST_truncated(datadir, train=True, download=is_download, transform=train_transform)
        fashionMNIST_test_ds = FashionMNIST_truncated(datadir, train=False, download=is_download, transform=test_transform)

    X_train, y_train = fashionMNIST_train_ds.data, fashionMNIST_train_ds.targets
    X_test, y_test = fashionMNIST_test_ds.data, fashionMNIST_test_ds.targets

    return (X_train, y_train, X_test, y_test, fashionMNIST_train_ds, fashionMNIST_test_ds)


def partition_data(dataset, datadir, partition, n_nets, alpha, process_id, synthetic_data_url, private_local_data):
    np.random.seed(10)
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test, fashionMNIST_train_ds, fashionMNIST_test_ds = load_fashionMNIST_data(datadir, process_id, synthetic_data_url, private_local_data)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fix":
        dataidx_map_file_path = "./data_preprocessing/non-iid-distribution/FashionMNIST/net_dataidx_map.txt"
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = "./data_preprocessing/non-iid-distribution/FashionMNIST/distribution.txt"
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, fashionMNIST_train_ds, fashionMNIST_test_ds


# for centralized training
def get_dataloader(
    dataset,
    datadir,
    train_bs,
    test_bs,
    dataidxs=None,
    data_efficient_load=False,
    full_train_dataset=None,
    full_test_dataset=None,
):
    return get_dataloader_FashionMNIST(
        datadir,
        train_bs,
        test_bs,
        dataidxs,
        data_efficient_load=data_efficient_load,
        full_train_dataset=full_train_dataset,
        full_test_dataset=full_test_dataset,
    )


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_FashionMNIST(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_FashionMNIST(
    datadir,
    train_bs,
    test_bs,
    dataidxs=None,
    data_efficient_load=False,
    full_train_dataset=None,
    full_test_dataset=None,
):
    transform_train, transform_test = _data_transforms_fashionMNIST()

    if data_efficient_load:
        dl_obj = FashionMNIST_truncated_WO_reload
        train_ds = dl_obj(
            datadir, dataidxs=dataidxs, train=True, transform=transform_train, full_dataset=full_train_dataset
        )
        test_ds = dl_obj(datadir, train=False, transform=transform_test, full_dataset=full_test_dataset)
    else:
        dl_obj = FashionMNIST_truncated
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_test_FashionMNIST(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = FashionMNIST_truncated

    transform_train, transform_test = _data_transforms_fashionMNIST()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True,)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True,)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def load_partition_data_distributed_fashionMNIST(
    process_id,
    dataset,
    data_dir,
    partition_method,
    partition_alpha,
    client_number,
    batch_size,
    data_efficient_load=True,
):
    (
        X_train,
        y_train,
        X_test,
        y_test,
        net_dataidx_map,
        traindata_cls_counts,
        cifar10_train_ds,
        cifar10_test_ds,
    ) = partition_data(dataset, data_dir, partition_method, client_number, partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(
            dataset,
            data_dir,
            batch_size,
            batch_size,
            data_efficient_load=True,
            full_train_dataset=cifar10_train_ds,
            full_test_dataset=cifar10_test_ds,
        )
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(
            dataset,
            data_dir,
            batch_size,
            batch_size,
            dataidxs,
            data_efficient_load=True,
            full_train_dataset=cifar10_train_ds,
            full_test_dataset=cifar10_test_ds,
        )
        logging.info(
            "process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d"
            % (process_id, len(train_data_local), len(test_data_local))
        )
        train_data_global = None
        test_data_global = None
    return (
        train_data_num,
        train_data_global,
        test_data_global,
        local_data_num,
        train_data_local,
        test_data_local,
        class_num,
    )


def efficient_load_partition_data_fashionMNIST(
    args,
    dataset,
    data_dir,
    partition_method,
    partition_alpha,
    client_number,
    batch_size,
    process_id=0,
    synthetic_data_url="",
    private_local_data="",
    n_proc_in_silo=0,
    data_efficient_load=True,
):
    (
        X_train,
        y_train,
        X_test,
        y_test,
        net_dataidx_map,
        traindata_cls_counts,
        cifar10_train_ds,
        cifar10_test_ds,
    ) = partition_data(dataset, data_dir, partition_method, client_number, partition_alpha, process_id, synthetic_data_url, private_local_data)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(
        dataset,
        data_dir,
        batch_size,
        batch_size,
        data_efficient_load=True,
        full_train_dataset=cifar10_train_ds,
        full_test_dataset=cifar10_test_ds,
    )
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(
            dataset,
            data_dir,
            batch_size,
            batch_size,
            dataidxs,
            data_efficient_load=True,
            full_train_dataset=cifar10_train_ds,
            full_test_dataset=cifar10_test_ds,
        )
        logging.info(
            "client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d"
            % (client_idx, len(train_data_local), len(test_data_local))
        )
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    # NIID可控逻辑

    # 对客户端按数据点数降序排序
    sorted_clients_idx = sorted(data_local_num_dict, key=data_local_num_dict.get, reverse=True)

    # 计算x%的数据的索引
    x_partition_idx = round(len(sorted_clients_idx) * args.experiment_niid_level)

    # 将客户端分为两部分
    x_sorted_clients = sorted_clients_idx[:x_partition_idx]
    remaining_clients = sorted_clients_idx[x_partition_idx:]

    # 打乱剩余客户端的顺序
    np.random.shuffle(remaining_clients)

    # 将两部分客户端合并
    mixed_clients = np.concatenate((x_sorted_clients, remaining_clients), axis=0)

    # 为每个客户端分配训练和测试数据
    train_data_local_dict_mixed = dict()
    test_data_local_dict_mixed = dict()

    idx = 0
    for client_idx in mixed_clients:
        train_data_local_dict_mixed[idx] = train_data_local_dict[client_idx]
        test_data_local_dict_mixed[idx] = test_data_local_dict[client_idx]
        idx += 1

    # NIID可控逻辑^^^^

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict_mixed,
        test_data_local_dict_mixed,
        class_num,
    )
