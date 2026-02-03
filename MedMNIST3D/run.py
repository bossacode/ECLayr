import sys
sys.path.append("../")
import torch
import pandas as pd
# import wandb
import argparse
import yaml
import os
from train_eval import run, run_wandb


# for reproducibility (may degrade performance)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(123)


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(description='RUN Baseline model of MedMNIST3D')
    parser.add_argument('--data_flag', default='organmnist3d', type=str)
    parser.add_argument('--conv', default='ACSConv', help='choose converter from Conv2_5d, Conv3d, ACSConv', type=str)
    parser.add_argument('--model_flag', default='resnet18', help='choose backbone, resnet18/resnet50', type=str)
    parser.add_argument('--shape_transform', help='for shape dataset, whether multiply 0.5 at eval', action="store_true")
    args = parser.parse_args()
    data_flag = args.data_flag
    conv = args.conv
    model_flag = args.model_flag
    shape_transform = args.shape_transform

    # load configuration file needed for training model
    with open(f"configs/{args.data_flag}/{args.model_flag}.yaml", "r") as f:
        cfg = yaml.load(f, yaml.FullLoader)

    # add device to configuration file
    if torch.cuda.is_available():
        cfg["device"] = "cuda"
    elif torch.backends.mps.is_available():
        cfg["device"] = "mps"
    else:
        cfg["device"] = "cpu"

    nsim = 10    # number of simulations to run

    result_dir = f"./results/{data_flag}"    # directory to save results
    os.makedirs(result_dir, exist_ok=True)
    
    # wandb.login()

    print(f"Training {model_flag} model...")
    print(cfg)
    history = []
    # loop over number of simulations
    for sim in range(1, nsim+1):
        print("\n")
        print("-"*60)
        print(f"Simulation: [{sim} / {nsim}]")
        print("-"*60)
        
        test_acc, train_acc, runtime, best_epoch = run(data_flag, cfg, conv, model_flag, shape_transform)
        # run_wandb(data_flag, cfg, conv, model_flag, shape_transform, name=f"sim{sim}")
        history.append((train_acc, test_acc, runtime, best_epoch))

    df = pd.DataFrame(history, columns=["Train Accuracy", "Test Accuracy", "Runtime", "Best Epoch"])
    df.to_csv(f"{result_dir}/{args.model_flag}.csv", index=False)