import numpy as np
from tadasets.dimension import embed
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import os
from sklearn.model_selection import train_test_split

def dsphere(n=100, d=2, r=1, noise=None, ambient=None):
    """
    Sample `n` data points on a d-sphere.

    Parameters
    -----------
    n : int
        Number of data points in shape.
    r : float
        Radius of sphere.
    ambient : int, default=None
        Embed the sphere into a space with ambient dimension equal to `ambient`. The sphere is randomly rotated in this high dimensional space.
    """
    data = np.random.randn(n, d+1)

    # Normalize points to the sphere
    data = r * data / np.sqrt(np.sum(data**2, 1)[:, None]) 

    if noise: 
        data += noise * np.random.randn(*data.shape)

    if ambient:
        assert ambient > d, "Must embed in higher dimensions"
        data = embed(data, ambient)
    return data


def create_sphere_dataset(n_samples=500, d=100, n_spheres=11, r=5, plot=False, seed=42):
    np.random.seed(seed)

    #it seemed that rescaling the shift variance by sqrt of d lets big sphere stay around the inner spheres
    variance=10/np.sqrt(d)

    shift_matrix = np.random.normal(0,variance,[n_spheres, d+1])

    spheres = [] 
    n_datapoints = 0
    for i in np.arange(n_spheres-1):
        sphere = dsphere(n=n_samples, d=d, r=r)
        spheres.append(sphere + shift_matrix[i,:])
        n_datapoints += n_samples

    #Additional big surrounding sphere:
    n_samples_big = 10*n_samples #int(n_samples/2)
    big = dsphere(n=n_samples_big, d=d, r=r*5)
    spheres.append(big)
    n_datapoints += n_samples_big

    if plot: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = mpl.cm.rainbow(np.linspace(0, 1, n_spheres))
        for data, color in zip(spheres, colors):
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=[color])
        plt.show()

    #Create Dataset:
    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints) 
    label_index=0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index:label_index + n_sphere_samples] = index
        label_index += n_sphere_samples
    
    return dataset, labels


if __name__ == "__main__":
    X, y = create_sphere_dataset(n_samples=500, d=100, n_spheres=11, r=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

    os.makedirs("./data/", exist_ok=True)
    torch.save((torch.from_numpy(X_tr).to(torch.float32), torch.from_numpy(y_tr).to(torch.float32)), "./data/train.pt")
    torch.save((torch.from_numpy(X_val).to(torch.float32), torch.from_numpy(y_val).to(torch.float32)), "./data/val.pt")
    torch.save((torch.from_numpy(X_test).to(torch.float32), torch.from_numpy(y_test).to(torch.float32)), "./data/test.pt")