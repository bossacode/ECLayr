import sys
sys.path.append("../")
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
import numpy as np
import os
from utils.preprocess import corrupt_noise, dtm_transform


def gen_noise_data(cn_prob, train_samples=1000, val_size=0.3):
    """Sample and generate MNIST data that are contaminated by corruption and noise.

    Args:
        cn_prob (list): List containing corruption and noise probabilities.
        train_samples (int, optional): Number of train data to sample. Defaults to 1000.
        val_size (float, optional): Proportion of validation split. Defaults to 0.3.
    """
    train_data = MNIST(root="./dataset/raw/", train=True, download=True, transform=ToTensor())  # shape: (60000, 28, 28)
    test_data = MNIST(root="./dataset/raw/", train=False, download=True, transform=ToTensor())  # shape: (10000, 28, 28)
    
    # normalize and add channel dimension: (N, 1, H, W)
    x_train = (train_data.data / 255).unsqueeze(1)
    y_train = train_data.targets
    x_test = (test_data.data / 255).unsqueeze(1)
    y_test = test_data.targets

    # sample "train_samples" number of train data with equal proportion per label
    train_idx = []
    labels = y_train.unique()
    num_labels = len(labels)
    for i in labels:
        idx = torch.where(y_train == i)[0]
        sampled_idx = np.random.choice(idx, size=int(train_samples / num_labels), replace=False)
        train_idx.extend(sampled_idx)
    x_tr_sampled, y_tr_sampled = x_train[train_idx], y_train[train_idx]

    # split train and validation data
    x_tr, x_val, y_tr, y_val = train_test_split(x_tr_sampled, y_tr_sampled, test_size=val_size, random_state=42, shuffle=True, stratify=y_tr_sampled)

    for p in cn_prob:
        # contaminate data
        x_tr_cn = corrupt_noise(x_tr, p)
        x_val_cn = corrupt_noise(x_val, p)
        x_test_cn = corrupt_noise(x_test, p)

        # apply DTM
        x_tr_cn_dtm = dtm_transform(x_tr_cn, m0=0.05, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])        
        x_val_cn_dtm = dtm_transform(x_val_cn, m0=0.05, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])
        x_test_cn_dtm = dtm_transform(x_test_cn, m0=0.05, lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28])

        # save train, validation and test data
        dir = "./dataset/processed/" + str(int(p * 100)).zfill(2)
        os.makedirs(dir)
        torch.save((x_tr_cn, x_tr_cn_dtm, y_tr), f=f"{dir}/train.pt")
        torch.save((x_val_cn, x_val_cn_dtm, y_val), f=f"{dir}/val.pt")
        torch.save((x_test_cn, x_test_cn_dtm, y_test), f=f"{dir}/test.pt")


if __name__ == "__main__":
    cn_prob = [0.0, 0.05, 0.1, 0.15, 0.2]   # corruption and noise probabilities
    train_samples = 1000                    # number of train data to sample
    val_size = 0.3                          # proportion of validation split

    np.random.seed(42)
    torch.manual_seed(42)
    gen_noise_data(cn_prob, train_samples, val_size)