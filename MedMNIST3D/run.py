import torch
import wandb
import argparse
import yaml
from train_eval import run, run_wandb


# for reproducibility (may degrade performance)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(123)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST3D')

    parser.add_argument('--data_flag',
                        default='organmnist3d',
                        type=str)
    
    parser.add_argument('--conv',
                        default='ACSConv',
                        help='choose converter from Conv2_5d, Conv3d, ACSConv',
                        type=str)
    
    parser.add_argument('--model_flag',
                        default='resnet18',
                        help='choose backbone, resnet18/resnet50',
                        type=str)
    
    parser.add_argument('--shape_transform',
                        help='for shape dataset, whether multiply 0.5 at eval',
                        action="store_true")


    args = parser.parse_args()
    data_flag = args.data_flag
    conv = args.conv
    model_flag = args.model_flag
    shape_transform = args.shape_transform

    # load configuration file needed for training model
    with open(f"configs/{args.data_flag}/{args.model_flag}.yaml", "r") as f:
        cfg = yaml.load(f, yaml.FullLoader)
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"  # add device to configuration file

    nsim = 10    # number of simulations to run
    wandb.login()

    print(shape_transform)

    # loop over number of simulations
    for sim in range(1, nsim+1):
        print(f"\nSimulation: [{sim} / {nsim}]")
        print("-"*30)
        name = f"sim{sim}"  # used for specifying runs in wandb
        run_wandb(data_flag, cfg, conv, model_flag, shape_transform, name)
        # run(data_flag, cfg, conv, model_flag, shape_transform)