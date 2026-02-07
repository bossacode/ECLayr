# !/bin/bash

python run.py -m Cnn
python run.py -m ECCnn_i
python run.py -m ECCnn
python run.py -m DECCnn

cd ../ph_models

python run_mnist.py -m PersCnn
python run_mnist.py -m PLCnn_i
python run_mnist.py -m PLCnn