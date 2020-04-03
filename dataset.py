import os
import pickle
import torch
from torch.utils.data import DataLoader, Dataset


class Simple_Dataset(Dataset):
    def __init__(self, stage, data, label):
        self.stage = stage
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.size()[0]


def read_train_test_nn(positive_train, positive_test, negative_train, negative_test, bs):
    # number of train and test
    num_of_train, num_of_test = len(positive_train), len(positive_test)
    num_of_negative_train, num_of_negative_test = len(negative_train), len(negative_test)
    # data and label
    train_data = positive_train + negative_train
    train_label = [0 for _ in range(num_of_train)] + [1 for _ in range(num_of_negative_train)]
    test_data = positive_test + negative_test
    test_label = [0 for _ in range(num_of_test)] + [1 for _ in range(num_of_negative_test)]
    # data set and data loader
    train_set = Simple_Dataset(stage="train", data=torch.LongTensor(train_data), label=torch.LongTensor(train_label))
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
    test_set = Simple_Dataset(stage="test", data=torch.LongTensor(test_data), label=torch.LongTensor(test_label))
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
    # use data loader
    return train_loader, test_loader


def read_train_test_dm(positive_train, positive_test, negative_train, negative_test, bs):
    # number of train and test
    num_of_train, num_of_test = len(positive_train), len(positive_test)
    num_of_negative_train, num_of_negative_test = len(negative_train), len(negative_test)
    # data and label
    train_data = positive_train + negative_train
    train_label = [1 for _ in range(num_of_train)] + [0 for _ in range(num_of_negative_train)]
    test_data = positive_test + negative_test
    test_label = [1 for _ in range(num_of_test)] + [0 for _ in range(num_of_negative_test)]
    # data set and data loader
    train_set = Simple_Dataset(stage="train", data=torch.LongTensor(train_data), label=torch.FloatTensor(train_label))
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
    test_set = Simple_Dataset(stage="test", data=torch.LongTensor(test_data), label=torch.FloatTensor(test_label))
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
    # use data loader
    return train_loader, test_loader
