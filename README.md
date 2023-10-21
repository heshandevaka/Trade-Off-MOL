# Trade off in Multi-objective Learning

This repository contains the code for experiments of the paper: ["Three-Way Trade-Off in Multi-Objective Learning: Optimization, Generalization and Conflict-Avoidance"](https://arxiv.org/pdf/2305.20057.pdf).

In this work, we study the optimization, generalization, and conflict avoidance in stochastic multi-objective learning (MOL), with an instantiation of the proposed Multi-objective gradient with Double sampling (MoDo) algorithm.

MoDo is a variant of the stochastic MGDA method, with double sampling to mitigate gradient bias.

<p align="center">
<img width="800" alt="desc_space" src="https://github.com/heshandevaka/Trade-Off-MOL/assets/96305785/b84fdf81-2e95-479f-b874-c5394af34d50">
</p>

<p align="center">
<img width="750" alt="desc_space" src="https://github.com/heshandevaka/Trade-Off-MOL/assets/96305785/c75cb5cd-2df6-4cb0-8dde-9733b1452cfb)">
</p>





# Environment setup

1. Use the following command to install the dependencies
```
conda create -n moo python=3.8
conda activate moo
conda install pytorch torchvision==0.9.0 torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install numpy scipy seaborn tqdm
conda install -c conda-forge cvxpy
```
2. Install LibMTL

```
cd LibMTL
pip install -e .
```

# Experiments

## LiBMTL

This folder contains the framework to run complex/real world MTL tasks from [LibMTL](https://github.com/median-research-group/LibMTL). The framework contains five tasks: nyu, office-31, office-home, qm9, xtreme datasets. Currently, MoDo and MoCo is implemented for office-31 and office-home tasks. The code to run each task can be found in directory `LibMTL/LiBMTL/examples/<Task>`. The specifics on how to run are described in the README in respective task folders. 

To run an experiment with MoDo, navigate to `LibMTL/LiBMTL/examples/office`, and run

```shell
python train_office.py --multi_input --dataset_path office-31 --weighting MoDo
```

To check the hyper-parameters and methods allowed (e.g. MoDo), run

```shell
python train_office.py -h
```

## Toy-MNIST

This folder contains the toy MOO task designed using multiple loss functions applied to a multi-layer perceptron (MLP) for learning to classify the MNIST dataset. The code structure is simpler compared to the LibMTL framework, although similar in implementation. Currently the code is designed to use three loss functions: cross-entropy loss, mean square error (MSE) loss, and Huber loss. The training logs of this task report overall classification accuracy, per objective loss, and Pareto stationarity measure for train, test and validation datasets.

To run an experiment with MoDo, navigate to `Toy-MNIST`, and run

```shell
python toy.py --moo_method MoDo
```

To check the hyper-parameters and methods allowed (e.g. MoDo), run

```shell
python toy.py -h
```

## Toy

This folder contains the toy MOO task designed using explicitly defining two objectives $f_1(x)$ and $f_2(x)$, which is inspired by prior work [CAGrad](https://github.com/Cranial-XIX/CAGrad). To create a mock empirical data set, we sample $n$ data points $\{z_i\}$ from a zero mean normal distribution. We set the population objectives to be the original objectives $f_1(x)$ and $f_2(x)$, and construct the corresponding empirical datasets as $f_1(x) +  \bar{z}^\top x$ and $f_2(x) + \bar{z}^\top x $, respectively, where $\bar{z}$ is the mean of the datapoints $z_i$. A stochastic sample of the objectives are then defined as $f_1(x) +  z_i^\top x$ and $f_2(x) + z_i^\top x $, for some $i\in[n]$. Experimanets are run to see how well the algorithms perform in the stochastic setting in terms of achieving Pareto optimality with respect to the population objective.

To run an experiment with the toy objectives, navigate to `Toy`, and run

```shell
python toy-stochastic.py
```

## License

MIT license

## Citation

```
@inproceedings{chen2023three,
  title={Three-Way Trade-Off in Multi-Objective Learning: Optimization, Generalization and Conflict-Avoidance},
  author={Chen, Lisha and Fernando, Heshan and Ying, Yiming and Chen, Tianyi},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```


## Ackowledgement

- The Multi-Task Learning (MTL) benchmark experiments use [LibMTL](https://github.com/median-research-group/LibMTL).
- The nonconvex toy experiment is modified from the toy example in [CAGrad](https://github.com/Cranial-XIX/CAGrad).

We thank the authors for providing the code and data. Please cite their works and ours if you use the code or data.
