# DynPredCode
Code acommpanying _Dynamic Predictive Coding: A Model of Hierarchical Sequence
Learning and Prediction in the Neocortex_

For two-level results, stay on this branch (`main`). For three-level results, switch to the `three_level` branch.

## Setup
We provide an [Apptainer](https://apptainer.org/) image `predcode.sif` (similar to a Docker image), which provides all the necessary dependencies to train and analyze the model. You can download the image [here](https://drive.google.com/drive/folders/120T-wChXIR-aI7zL-9AeQF1hq-5GZFIy?usp=sharing). You're also welcome to re-build this image by
```
apptainer build --bind $PWD:/mnt --fakeroot predcode.sif apptainer.def
```
## Data
You can download the data [here](https://drive.google.com/drive/folders/120T-wChXIR-aI7zL-9AeQF1hq-5GZFIy?usp=sharing). The data folder should be placed in the root directory of this repository.

The data folder for three-level models are:
- `data/clock`: MNIST data with clockwise bouncing type
- `data/straight`: MNIST data with straight bouncing type
- `data/three`: `clock` and `straight` combined

## Training the models

You can download the `experiments` folder from [here](https://drive.google.com/drive/folders/120T-wChXIR-aI7zL-9AeQF1hq-5GZFIy?usp=sharing), which contains pretrained model weights. It also contains the `parmas.json` files which can be used to train the models (see below). The experiments folder should be placed in the root directory of this repository.

### Pretraining the second-level transition matrices
Clockwise bouncing:
```
apptainer run --home $PWD --nv predcode.sif python train_two_trans.py --model_dir experiments/two_trans_clock --data_dir data/clock
```

Straight bouncing:
```
apptainer run --home $PWD --nv predcode.sif python train_two_trans.py --model_dir experiments/two_trans_straight --data_dir data/straight
```

### Training the three-level model

```
apptainer run --home $PWD --nv predcode.sif python train_three.py --model_dir experiments/three --data_dir data/three
```

## Running the analysis
The `analysis` folder contains two notebooks:
- `analysis/err_dist.ipynb`: analysis for calculating the first-level prediction error threshold
- `analysis/viz.ipynb`: analysis for the three-level model
