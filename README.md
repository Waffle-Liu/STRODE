# Stochastic Boundary Ordinary Differential Equation (STRODE)
This is the authors' official PyTorch implementation for STRODE. This repo contains code for experiments in the **ICML 2021** paper '[Stochastic Boundary Ordinary Differential Equation]()'.

## Project Description

The precise timing of each item in real-world data streams can carry important information about the underlying dynamics.  However, most algorithms for time-series modeling, e.g., latent ordinary differential equation (ODE), fail to infer timings directly from visual or audio inputs, and still require a large amount of training data with timing annotations. Inspired by neuroscience perspectives on time perception, we generalize neural ODE in handling a special type of boundary value problem with random boundary times and propose a probabilistic ordinary differential equation (ODE), called Stochastic boundaRy ODE (STRODE). Extensive experiments show that STRODE learns both the timings and the dynamics of time series data without requiring any timing annotations during training.

## How to run
### Environment
* Python 3.7
* PyTorch 1.7.1

### Running experiments

#### Toy Dataset

For training and evaluation on the toy dataset, run the following script:
```
cd toy
python run.py --dataset exp
```
where `--dataset` specify the dataset used in the experiments. For more specifications of the experiments, see details in `config.py`.

Run the following script to draw the figure
```
python draw.py
```

#### Rotating MNIST ThumbnailT 

For training and evaluation of STRODE, run the following script:
```
cd rotateMNIST
python run.py --model strode --dataset exp
```
where `--model` and `--dataset` specify the model and the dataset used in the experiments. For more specifications of the experiments, see details in `config.py`.


### Data

You can directly use the datasets we generate in `./data` by setting `isLoad=True`(default) in `config.py`. You can also set `isLoad=False` and run the script above(new datasets will be automatically generated)before training the model. If you run the RotateMNIST experiment, make sure that you download the MNIST dataset. You can directly use `torchvision`. 


## Citation
If you use STRODE or this codebase in your own work, please cite our paper: 
```

```

## License
MIT
