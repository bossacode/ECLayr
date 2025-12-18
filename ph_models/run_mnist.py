import tensorflow as tf
import wandb
import argparse
import yaml
from train_tf import run, run_wandb
from ph_models.mnist_models import PersCnn, PLCnn_i, PLCnn


# for reproducibility (may degrade performance)
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()


parser = argparse.ArgumentParser()
parser.add_argument("--model_flag", help="Name of model to train")
args = parser.parse_args()
# load configuration file needed for training model
with open(f"configs/MNIST/{args.model_flag}.yaml", "r") as f:
    cfg = yaml.load(f, yaml.FullLoader)


models = {
    "PersCnn": PersCnn,
    "PLCnn_i": PLCnn_i,
    "PLCnn": PLCnn
    }


if __name__ == "__main__":
    nsim = 15                               # number of simulations to run
    cn_prob = [0.0, 0.05, 0.1, 0.15, 0.2]   # corruption and noise probabilities

    wandb.login()
    project = "MNIST"           # used as project name in wandb
    group = args.model_flag     # used for grouping experiments in wandb

    # loop over different noise probability
    for p in cn_prob:
        prob = str(int(p * 100)).zfill(2)
        job_type = prob                                     # used for grouping experiments in wandb
        data_dir = f"../MNIST/dataset/processed/{prob}/"    # base directory path to where data is loaded

        print("-"*30)
        print(f"Corruption & noise rate: {p}")
        print("-"*30)
        
        # loop over number of simulations
        for sim in range(1, nsim+1):
            print(f"\nSimulation: [{sim} / {nsim}]")
            print("-"*30)

            name = f"sim{sim}"  # used for specifying runs in wandb
            model = models[args.model_flag](**cfg["model_params"])
            
            run_wandb(model, cfg, data_dir, project, group, job_type, name)
            # run(model, cfg, data_dir)