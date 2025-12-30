import sys
sys.path.append("../")
import torch
import wandb
import argparse
import yaml
from utils.train_eval import run, run_wandb
from models import Cnn, ECCnn_i, ECCnn, DECCnn


# for reproducibility (may degrade performance)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)

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


if __name__ == "__main__":
    nsim = 15                                   # number of simulations to run
    noise_prob = [0.0, 0.05, 0.1, 0.15, 0.2]    # noise probabilities

    wandb.login()
    project = "ORBIT"           # used as project name in wandb
    group = args.model_flag     # used for grouping experiments in wandb

    for p in noise_prob:
        prob = str(int(p * 100)).zfill(2)
        job_type = prob                     # used for grouping experiments in wandb
        data_dir = f"./dataset/{prob}/"     # base directory path to where data is loaded
        
        print("-"*30)
        print(f"Corruption & noise rate: {p}")
        print("-"*30)
        
        # loop over number of simulations
        for sim in range(1, nsim+1):
            print(f"\nSimulation: [{sim} / {nsim}]")
            print("-"*30)
            
            name = f"sim{sim}"  # used for specifying runs in wandb
            model = model_dict[args.model_flag](device=cfg["device"], **cfg["model_params"]).to(cfg["device"])
            
            run_wandb(model, cfg, data_dir, project, group, job_type, name)
            # run(model, cfg, data_dir)