import sys
sys.path.append("../")
import torch
import pandas as pd
# import wandb
import argparse
import yaml
import os
from utils.train_eval import run, run_wandb
from models import Cnn, ECCnn_i, ECCnn, DECCnn


# for reproducibility (may degrade performance)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_flag", help="Name of model to train")
    args = parser.parse_args()

    # load configuration file needed for training model
    with open(f"configs/{args.model_flag}.yaml", "r") as f:
        cfg = yaml.load(f, yaml.FullLoader)

    # add device to configuration file
    if torch.cuda.is_available():
        cfg["device"] = "cuda"
    elif torch.backends.mps.is_available():
        cfg["device"] = "mps"
    else:
        cfg["device"] = "cpu" 

    model_dict = {
        "Cnn": Cnn,
        "ECCnn_i": ECCnn_i,
        "ECCnn": ECCnn,
        "DECCnn": DECCnn
        }

    nsim = 15                                   # number of simulations to run
    noise_prob = [0.0, 0.05, 0.1, 0.15, 0.2]    # noise probabilities
    
    result_dir = "./results"    # directory to save results
    os.makedirs(result_dir, exist_ok=True)

    # wandb.login()
    
    print(f"Training {args.model_flag} model...")
    print(cfg)
    history = []
    for p in noise_prob:
        prob = str(int(p * 100)).zfill(2)
        data_dir = f"./dataset/{prob}/"     # base directory path to where data is loaded
        
        # loop over number of simulations
        sim = 1
        while sim <= nsim:
            print("\n")
            print("-"*60)
            print(f"Corruption & noise rate: {p}")
            print(f"Simulation: [{sim} / {nsim}]")
            print("-"*60)
            
            model = model_dict[args.model_flag](device=cfg["device"], **cfg["model_params"]).to(cfg["device"])
            test_acc, train_acc, runtime, best_epoch = run(model, cfg, data_dir)
            # run_wandb(model, cfg, data_dir, project="ORBIT", group=args.model_flag, job_type=prob, name=f"sim{sim}")
            if train_acc < 80:
                print(f"Train accuracy {train_acc} is lower than 80%, repeating simulation {sim}...")
                continue
            history.append((int(p * 100), train_acc, test_acc, runtime, best_epoch))
            sim += 1

    df = pd.DataFrame(history, columns=["Job Type", "Train Accuracy", "Test Accuracy", "Runtime", "Best Epoch"])
    df.to_csv(f"{result_dir}/{args.model_flag}.csv", index=False)