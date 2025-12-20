# ECLayr: A Fast and Robust Topological Layer via Euler Characteristic Curves

## Requirements
Run the following commands to create a new conda virtual enviroment and download all necessary packages:
```
conda env create -f environment.yml
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

For cubical
```
cd ./eclayr/cubical/_ecc
python setup.py build_ext --inplace
```

For alpha
```
cd ./eclayr/alpha/cython_eclayr
python setup.py build_ext --inplace
```

For VR
```
cd .//eclayr/vr/cython_eclayr
python setup.py build_ext --inplace
```

## Run Experiments
To run dataset, move to directory. For example, to run mnist,
```
cd ./MNIST
python preprocess.py
python run_data.py args
python run_noise.py args
```
where args is the name of the model that can be found in the ```run_data.py``` or ```run_noise.py``` script.