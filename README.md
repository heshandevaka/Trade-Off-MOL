# Generalization in Multi-objective Optimization (MOO)

Experiments for generalization in MOO, and introducing MoDo, updated MoCo.

## LiBMTL

This folder contains the framework to run complex/real world MTL tasks. The framework contains five tasks: nyu, office-31, office-home, qm9, xtreme datasets. Currently, MoDo and MoCo is implemented for office-31 and office-home tasks. The code to run each task can be found in directory `LibMTL/LiBMTL/examples/<Task>`. The specifics on how to run are described in the README in respective task folders. 

To run an experiment with MoDo, navigate to `LibMTL/LiBMTL/examples/office`, and run

```shell
python train_office.py --multi_input --dataset_path office-31 --weighting MoDo
```

To check the hyper-parameters and methods allowed (e.g. MoCo, MoDo), run

```shell
python train_office.py -h
```

## Toy-MNIST

This folder contains the toy MOO task designed using multiple loss functions applied to a multi-layer perceptron (MLP) for learning to classify the MNIST dataset. The code structure is simpler compared to the LibMTL framework, althoogh similar in implementation. Currently the code is designed to use three loss functions: coress-entropy loss, l1-loss, and hinge-loss. The training logs of this task report overall classification accuracy, per objective loss, and Pareto stationarity measure for train, test and validation datasets.

To run an experiment with MoDo, navigate to `Toy-MNIST`, and run

```shell
python toy.py --moo_method MoDo
```

To check the hyper-parameters and methods allowed (e.g. MoCo, MoDo), run

```shell
python toy.py -h
```

## Toy

This folder contains the toy MOO task designed using explicitly defining two objectives $f_1(x)$ and $f_2(x)$. To create a mock empirical data set, we sample $n$ data points $\{z_i\}$ from a zero mean normal distribution. We set the population objectives to be the original objectives $f_1(x) and $f_2(x)$, and construct the corresponding empirical datasets as $f_1(x) +  \bar{z}^\top x$ and $f_2(x) + \bar{z}^\top x $, respectively, where $\bar{z}$ is the mean of the datapoints $z_i$. A stochastic sample of the objectives are then defined as $f_1(x) +  z_i^\top x$ and $f_2(x) + z_i^\top x $, for some $i\in[n]$. Experimanets are run to see how well the algorithms perform in the stochastic setting in terms of achieving Pareto optimality with respect to the population objective.

To run an experiment with the toy objectives, navigate to `Toy`, and run

```shell
python toy-stochastic.py
```
