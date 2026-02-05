import tensorflow as tf
import pandas as pd
# import wandb
import argparse
import yaml
import os
from train_tf import run, run_wandb
from ph_models.orbit_models import PersCnn, PLCnn_i, PLCnn


# for reproducibility (may degrade performance)
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_flag", help="Name of model to train")
    args = parser.parse_args()

    # load configuration file needed for training model
    with open(f"configs/ORBIT/{args.model_flag}.yaml", "r") as f:
        cfg = yaml.load(f, yaml.FullLoader)

    model_dict = {
        "PersCnn": PersCnn,
        "PLCnn_i": PLCnn_i,
        "PLCnn": PLCnn
        }

    nsim = 15                                   # number of simulations to run
    noise_prob = [0.0, 0.05, 0.1, 0.15, 0.2]    # noise probabilitieslities

    result_dir = "../ORBIT/results"             # directory to save results
    os.makedirs(result_dir, exist_ok=True)

    # wandb.login()
    
    print(f"Training {args.model_flag} model...")
    print(cfg)
    history = []
    for p in noise_prob:
        prob = str(int(p * 100)).zfill(2)
        data_dir = f"../ORBIT/dataset/{prob}/"  # base directory path to where data is loaded
        
        # loop over number of simulations
        sim = 1
        while sim <= nsim:
            print("\n")
            print("-"*60)
            print(f"Corruption & noise rate: {p}")
            print(f"Simulation: [{sim} / {nsim}]")
            print("-"*60)
            
            model = model_dict[args.model_flag](**cfg["model_params"])
            test_acc, train_acc, runtime, best_epoch = run(model, cfg, data_dir)
            # run_wandb(model, cfg, data_dir, project="ORBIT", group=args.model_flag, job_type=prob, name=f"sim{sim}")
            if train_acc < 80:
                print(f"Train accuracy {train_acc} is lower than 80%, repeating simulation {sim}...")
                continue
            history.append((int(p * 100), train_acc, test_acc, runtime, best_epoch))
            sim += 1

    df = pd.DataFrame(history, columns=["Job Type", "Train Accuracy", "Test Accuracy", "Runtime", "Best Epoch"])
    df.to_csv(f"{result_dir}/{args.model_flag}.csv", index=False)