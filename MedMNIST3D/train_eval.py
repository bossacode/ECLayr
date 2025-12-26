import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report
import medmnist
from time import time
from copy import deepcopy
from tqdm import trange
import wandb
from acsconv.converters import ACSConverter, Conv2_5dConverter, Conv3dConverter
from models import ResNet18, ECResNet18_i, ECResNet18, DECResNet18
from utils import Transform3D, model_to_syncbn, EarlyStopping


def train(model, dataloader, loss_fn, optim, device):
    data_size = len(dataloader.dataset)
    ma_loss, correct = 0, 0
    model.train()
    for batch_idx, (X, y) in enumerate(dataloader, 1):
        X, y = X.to(device), y.squeeze(1).to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        ma_loss += (loss.item() * len(y))   # bc. loss_fn predicts avg loss
        correct += (y_pred.argmax(1) == y).sum().item()

        loss.backward()
        optim.step()
        optim.zero_grad()

        if batch_idx % 10 == 1:
            print(f"Training loss: {loss.item():>7f} [{batch_idx*len(y):>3d}/{data_size:>3d}]")
    ma_loss /= data_size                    # moving average of loss over 1 epoch
    ma_acc = (correct / data_size) * 100    # moving average of accuracy over 1 epoch
    print(f"Train error:\n Accuracy: {ma_acc:>0.3f}%, Avg loss: {ma_loss:>7f} \n")
    return ma_loss, ma_acc


def test(model, dataloader, loss_fn, device):
    y_pred_list, y_true_list = [], []
    data_size = len(dataloader.dataset)
    avg_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.squeeze(1).to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            avg_loss += (loss.item() * len(y))
            correct += (y_pred.argmax(1) == y).sum().item()
            y_pred_list.append(y_pred.argmax(1))
            y_true_list.append(y)
    avg_loss /= data_size
    accuracy = (correct / data_size) * 100
    print(f"Validation/Test error:\n Accuracy: {(accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")

    predicted = torch.concat(y_pred_list).to("cpu")
    ground_truth = torch.concat(y_true_list).to("cpu")
    report = classification_report(ground_truth, predicted, zero_division="warn")
    print(report)
    return avg_loss, accuracy


def run(data_flag, cfg, conv, model_flag, shape_transform, use_wandb=False):
    info = medmnist.INFO[data_flag]
    n_channels = 3 if cfg["as_rgb"] else info['n_channels']
    n_classes = len(info['label'])
    device = cfg["device"]
    num_workers = 4

    # set dataset
    DataClass = getattr(medmnist, info['python_class'])
    train_transform = Transform3D(mul='random') if shape_transform else Transform3D()
    eval_transform = Transform3D(mul='0.5') if shape_transform else Transform3D()
    train_dataset = DataClass(split='train', transform=train_transform, download=True, root="./data/", as_rgb=cfg["as_rgb"], size=cfg["size"])
    val_dataset = DataClass(split='val', transform=eval_transform, download=True, root="./data/", as_rgb=cfg["as_rgb"], size=cfg["size"])
    test_dataset = DataClass(split='test', transform=eval_transform, download=True, root="./data/", as_rgb=cfg["as_rgb"], size=cfg["size"])
    
    # set dataloader
    train_dl = data.DataLoader(dataset=train_dataset,
                                batch_size=cfg["batch_size"],
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)
    val_dl = data.DataLoader(dataset=val_dataset,
                                batch_size=cfg["batch_size"],
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True)
    test_dl = data.DataLoader(dataset=test_dataset,
                                batch_size=cfg["batch_size"],
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True)

    # set model
    ######################################################################
    if model_flag == 'resnet18':
        model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'ecresnet18_i':
        model = ECResNet18_i(in_channels=n_channels, num_classes=n_classes, **cfg["model_params"])    
    elif model_flag == 'ecresnet18':
        model = ECResNet18(in_channels=n_channels, num_classes=n_classes, **cfg["model_params"])
    elif model_flag == 'decresnet18':
        model = DECResNet18(in_channels=n_channels, num_classes=n_classes, **cfg["model_params"])
    else:
        raise NotImplementedError
    ######################################################################
    if conv=='ACSConv':
        model = model_to_syncbn(ACSConverter(model))
    if conv=='Conv2_5d':
        model = model_to_syncbn(Conv2_5dConverter(model))
    if conv=='Conv3d':
        model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=None))
    model = model.to(device)

    # val_evaluator = medmnist.Evaluator(data_flag, 'val', size=cfg.size, root="./data/")
    # test_evaluator = medmnist.Evaluator(data_flag, 'test', size=cfg.size, root="./data/")

    loss_fn = nn.CrossEntropyLoss() # set loss function
    # set optimizer
    param_list = []
    for name, layer in model.named_children():
        if "eclayr" in name or "decc" in name:   # set lr for all eclayr and decc
            print("name: ",name, "lr: ", cfg["lr_topo"])
            param_list.append({"params": layer.parameters(), "lr": cfg["lr_topo"]})
        else:
            print("name: ",name, "lr: ", cfg["lr"])
            param_list.append({"params": layer.parameters(), "lr": cfg["lr"]})
    optim = Adam(param_list, lr=cfg["lr"])
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * cfg.epochs, 0.75 * cfg.epochs], gamma=cfg.gamma)
    # scheduler = ReduceLROnPlateau(optim, mode="min", factor=cfg["factor"], patience=cfg["sch_patience"], threshold=cfg["threshold"], verbose=True)
    es = EarlyStopping(cfg["es_patience"], cfg["threshold"], val_metric="loss") # set early stopping

    # best_auc = 0
    # best_loss = float('inf')
    # best_epoch = 0
    # best_model_state = deepcopy(model.state_dict())
    # patience = 0

    # train
    start = time()
    for i_epoch in trange(1, cfg["epochs"]+1):
        print(f"\nEpoch: [{i_epoch} / {cfg['epochs']}]")
        print("-"*30)
        
        # train_loss, train_acc = train(model, train_loader, task, loss_fn, optimizer, device)
        # val_loss, val_auc, val_acc = test(model, val_evaluator, val_loader, task, loss_fn, device, run)

        train_loss, train_acc = train(model, train_dl, loss_fn, optim, device)
        val_loss, val_acc = test(model, val_dl, loss_fn, device)
        
        # scheduler.step(val_loss)

        # early stopping
        stop, improvement = es.stop_training(val_loss, val_acc, i_epoch)
        if use_wandb:
            wandb.log({"train":{"loss":train_loss, "accuracy":train_acc},
                    "val":{"loss":val_loss, "accuracy":val_acc},
                    "best_val":{"loss":es.best_loss, "accuracy":es.best_acc}}, step=i_epoch)
        if stop or i_epoch == cfg["epochs"]:
            end = time()
            training_time = end - start
            print(f"\nTraining time: {training_time}\n")
            # torch.save(best_model_state, weight_path)   # save model weights
            if use_wandb:
                wandb.log({"training_time": training_time})
                wandb.log({"best_epoch": es.best_epoch})
            break
        elif improvement:
            best_model_state = deepcopy(model.state_dict())
            
        # if val_auc > best_auc:
        #     best_epoch = i_epoch
        #     best_loss = val_loss
        #     best_auc = val_auc
        #     best_acc = val_acc
        #     # best_model = deepcopy(model)
        #     best_model_state = deepcopy(model.state_dict())
        #     print(f'Current best AUC: {(best_auc * 100):>0.3f}%')
        #     print(f'Current best epoch: {best_epoch}')
        #     patience = 0
        # else:
        #     patience += 1

        # wandb.log({"train":{"loss":train_loss, "accuracy":train_acc},
        #         "val":{"loss":val_loss, "auc":val_auc, "accuracy":val_acc},
        #         "best_val":{"loss":best_loss, "auc":best_auc, "accuracy":best_acc}}, step=i_epoch)
        
        # if patience > 30:
        #     break

    # end = time()
    # training_time = end - start
    # print(f"\nTraining time: {training_time}\n")
    # wandb.log({"training_time": training_time})
    # wandb.log({"best_epoch": best_epoch})

    # test
    model.load_state_dict(best_model_state)
    test_loss, test_acc = test(model, test_dl, loss_fn, device)
    if use_wandb:
        wandb.log({"test":{"loss":test_loss, "accuracy":test_acc}})
    # test_loss, test_auc, test_acc = test(model, test_evaluator, test_dl, task, loss_fn, device, run)
    # wandb.log({"test":{"loss":test_loss, "auc":test_auc, "accuracy":test_acc}})


def run_wandb(data_flag, cfg, conv, model_flag, shape_transform, name):
    with wandb.init(config=cfg, project=data_flag + "_new", group=f"{model_flag}", name=name):
        cfg = wandb.config
        run(data_flag, cfg, conv, model_flag, shape_transform, use_wandb=True)