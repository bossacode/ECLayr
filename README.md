# ECLayr: A Fast and Robust Topological Layer via Euler Characteristic Curves


## Requirements
Run the following commands to create a new conda virtual enviroment and download all necessary packages:
```
conda env create -f environment.yaml
```

Activate virtual environment:
```
conda activate eclayr
```

Run
```
pip install -e.
```


## Build ECLayr

For cubical:
```
cd ./eclayr/cubical/_ecc
python setup.py build_ext --inplace
```

For Vietoris-Rips:
```
cd ./eclayr/vr/_ecc
python setup.py build_ext --inplace
```

For Alpha:
```
cd ./eclayr/alpha/_ecc
python setup.py build_ext --inplace
```

## Topological Autoencoder
First, move to the corresponding directory and generate the data.
```
cd ./TopoAE
python generate_data.py
```
Then, open ```main.ipynb``` and run the jupyter notebook.


## MNIST
First, move to the corresponding directory and generate the data.
```
cd ./MNIST
python generate_data.py
```
Then, run
```
sh run.sh
```
to run all models, or selectively choose from the commands in ```run.sh``` to run selected models.


## ORBIT5K
First, move to the corresponding directory and generate the data.
```
cd ./ORBIT
python generate_data.py
```
Then, run
```
sh run.sh
```
to run all models, or selectively choose from the commands in ```run.sh``` to run selected models.


## MedMNIST3D
Move to the corresponding directory and run ```run.sh```
```
cd ./MedMNIST3D
sh run.sh
```
to run all models and data, or selectively choose from the commands in ```run.sh``` to run selected models and data.