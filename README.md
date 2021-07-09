# Stochastic Boundary Ordinary Differential Equation (STRODE)
This is the authors' official PyTorch implementation for STRODE. This repo contains code for experiments in the **ICML 2021** paper '[Stochastic Boundary Ordinary Differential Equation]()'.

## Abstract

Perception of time from sequentially acquired sensory inputs is rooted in everyday behaviors of individual organisms. Yet, most algorithms for time-series modeling fail to learn dynamics of random event timings directly from visual or audio inputs, requiring timing annotations during training that are usually unavailable for  real-world applications. For instance, neuroscience perspectives on postdiction imply that there exist variable temporal ranges within which the incoming sensory inputs can affect the earlier perception, but such temporal ranges are mostly unannotated for real applications such as automatic speech recognition (ASR).
In this paper, we present a probabilistic ordinary differential equation (ODE), called STochastic boundaRy ODE (STRODE), that learns both the timings and the dynamics of time series data without requiring any timing annotations during training. STRODE allows the usage of differential equations to sample from the posterior point processes, efficiently and analytically. We further provide theoretical guarantees on the learning of STRODE. Our empirical results show that our approach successfully infers event timings of time series data.
Our method achieves competitive or superior performances compared to existing
state-of-the-art methods for both synthetic and real-world datasets. 

## How to run
### Environment
* Python 3.7
* PyTorch 1.7.1

### Running experiments

#### Toy 

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

#### RotateMNIST 

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