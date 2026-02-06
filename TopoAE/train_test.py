import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib as mpl
import matplotlib.pyplot as plt
import os


class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.x, self.y = torch.load(file_path)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class EarlyStopping:
    def __init__(self, patience, threshold):
        """_summary_

        Args:
            patience (_type_): _description_
            threshold (_type_): _description_
            val_metric (str, optional): _description_. Defaults to "loss".
        """
        self.patience = patience
        self.threshold = threshold
        self.count = 0
        self.best_loss, self.best_epoch = float("inf"), None

    def stop_training(self, val_loss, epoch):
        stop, improvement = True, True
        diff = (self.best_loss - val_loss)
        if diff > self.threshold:   # improvement needs to be above threshold 
            self.count = 0
            self.best_loss, self.best_epoch = val_loss, epoch
            return not stop, improvement
        else:
            self.count += 1
            if self.count > self.patience:  # stop training if no improvement for patience + 1 epochs
                print("-"*30)
                print(f"Best Epoch: {self.best_epoch}")
                print(f"Best Validation Loss: {self.best_loss:>8f}")
                print("-"*30)
                return stop, not improvement
            return not stop, not improvement
        

def train(model, dataloader, optimizer, device):
    """
    train for 1 epoch
    """
    data_size = len(dataloader.dataset)
    ma_loss, ma_recon_loss, ma_topo_loss = 0, 0, 0
    model.train()
    for batch, (X, y) in enumerate(dataloader, 1): # X is a list containing batch of original data and DTM transformed data  
        X, y = X.to(device), y.to(device)
        X_recon, z, loss, (recon_loss, topo_loss) = model(X)
        ma_loss += (loss.item() * len(y))  # bc. loss_fn predicts avg loss
        ma_recon_loss += (recon_loss * len(y))
        ma_topo_loss += (topo_loss * len(y))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 1:
            print(f"Training loss: {loss.item():>7f} Recon loss: {recon_loss:>7f} Topo loss: {topo_loss:>7f} [{batch*len(y):>3d}/{data_size:>3d}]\n")
    ma_loss /= data_size    # moving average of loss over 1 epoch
    ma_recon_loss /= data_size
    ma_topo_loss /= data_size
    print(f"Train error:\n Avg loss: {ma_loss:>8f} Avg recon loss: {ma_recon_loss:>8f} Avg topo loss: {ma_topo_loss:>8f}\n")
    return ma_loss


def eval(model, dataloader, device, plot_latent=False):
    """
    """
    data_size = len(dataloader.dataset)
    avg_loss, avg_recon_loss, avg_topo_loss= 0, 0, 0
    z_list, y_list = [], []
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X_recon, z, loss, (recon_loss, topo_loss) = model(X)
            avg_loss += (loss.item() * len(y))
            avg_recon_loss += (recon_loss * len(y))
            avg_topo_loss += (topo_loss * len(y))
            if plot_latent:
                z_list.append(z)
                y_list.append(y)
    avg_loss /= data_size
    avg_recon_loss /= data_size
    avg_topo_loss /= data_size
    print(f"Validation/Test error:\n Avg loss: {avg_loss:>8f} Avg recon loss: {avg_recon_loss:>8f} Avg topo loss: {avg_topo_loss:>8f}\n")
    if plot_latent:
        latent = torch.concat(z_list, dim=0)
        labels = torch.concat(y_list, dim=0)
        plt.figure(figsize=(5, 5))
        plt.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        plt.scatter(latent[:, 0], latent[:, 1], s=2, c=labels, cmap=mpl.cm.viridis)
        plt.show();
    return avg_loss


def train_val(model, batch_size, lr, epochs, plot_latent=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = CustomDataset("./data/train.pt")
    val_dataset = CustomDataset("./data/val.pt")
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size)

    optim = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    es = EarlyStopping(patience=10, threshold=0.0001)
    
    for n_epoch in range(1, epochs+1):
        print(f"\nEpoch: [{n_epoch} / {epochs}]")
        print("-"*30)

        train_loss = train(model, train_dataloader, optim, device)
        val_loss = eval(model, val_dataloader, device, plot_latent)

        # early stopping
        stop, improvement = es.stop_training(val_loss, n_epoch)
        if stop:
            os.makedirs("./saved_weights/", exist_ok=True)
            torch.save(model_state_dict, f"./saved_weights/{model.__class__.__name__}.pt")  # save model weights
            break
        elif improvement:
            model_state_dict = model.state_dict()


def test(model, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dataset = CustomDataset("./data/test.pt")
    test_dataloader = DataLoader(test_dataset, batch_size)
    test_loss = eval(model, test_dataloader, device, plot_latent=True)