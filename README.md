# DSTLchallenge
Completing this challenge as a final project for the Deep Learning course at the University of Trieste, summer 2025.

# Setup 
This setup is quite involved

First of all, create the necessary conda environment with all the packages, 
```
conda env create --file env-cpu.yml -n dlprojenv
```

# Training
Before actually training the model, we have to be able to preprocess the images. We adapted [this approach](https://www.kaggle.com/code/kuklaolga/end-to-end-baseline-with-u-net-keras) to pytorch.
In order to preprocess data from scratch, run:
```bash
python3 -m src.utils.format_data
```
