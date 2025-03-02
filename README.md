# DynPredCode
Code accompanying [_Dynamic Predictive Coding: A Model of Hierarchical Sequence
Learning and Prediction in the Neocortex_](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011801)

For two-level results, stay on this branch (`main`). For three-level results, switch to the `three_level` branch.

## Setup
We provide an [Apptainer](https://apptainer.org/) image `predcode.sif` (similar to a Docker image), which provides all the necessary dependencies to train and analyze the model. You can download the image [here](https://drive.google.com/drive/folders/120T-wChXIR-aI7zL-9AeQF1hq-5GZFIy?usp=sharing). You're also welcome to re-build this image by
```
apptainer build --bind $PWD:/mnt --fakeroot predcode.sif apptainer.def
```
## Data
You can download the data [here](https://drive.google.com/drive/folders/120T-wChXIR-aI7zL-9AeQF1hq-5GZFIy?usp=sharing). The data folder should be placed in the root directory of this repository.

## Training the models

You can download the `experiments` folder from [here](https://drive.google.com/drive/folders/120T-wChXIR-aI7zL-9AeQF1hq-5GZFIy?usp=sharing), which contains pretrained model weights. It also contains the `parmas.json` files which can be used to train the models (see below). The experiments folder should be placed in the root directory of this repository.

To train the model using the natural video dataset: 
```
apptainer run --home $PWD --nv predcode.sif python train_ista.py --model_dir experiments/two_forest --data_dir data/forest
```

To train the model using the Moving MNIST dataset:
```
apptainer run --home $PWD --nv predcode.sif python train_ista.py --model_dir experiments/two_mnist --data_dir data/mnist
```

You will see that **we have already provided pretrained models** in the `experiments` directory. In either folder, we used the model at the last iteration, `last.pth.tar`.

### Training the memory model

To train the memory model: 
```
apptainer run --home $PWD --nv predcode.sif python train_memory.py --model_dir experiments/memory --data_dir data/memory
```

To train the memory model with multiple conditioning:
```
apptainer run --home $PWD --nv predcode.sif python train_memory.py --model_dir experiments/memory_multi --data_dir data/memory_multi
```

## Running the analysis
All analysis scripts are saved in the `analysis` folder. To run any of the analysis:
```
cd analysis/
apptainer run --home $PWD --nv predcode.sif python [xxx_analysis.py]
```

Each of these analysis will save analysis metadata in `analysis/results` folder. For any script, add the `-h` flag to see all options.

## Visualizing the results
Once all analysis scripts have been run, you can run through the `visualization.ipynb` notebook to visualize the results and there should be no errors. 

The figures produced by the visualization notebook are saved in `analysis/figures` folder, which make up all of the figures (except for the schematics) in the paper.
