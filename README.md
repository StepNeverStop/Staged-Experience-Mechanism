# Staged Experience Mechanism (SEM)

This repository is the source code of paper *SEM: Adaptive Staged Experience Access Mechanism for Reinforcement Learning* of **ICTAI 2020**.

## Getting started

```python
"""
Usage:
    python [options]

Options:
    -h,--help                   Help
    -i,--inference              Inference mode [default: False]
    -a,--algorithm=<name>       Specify training algorithm [default: ppo]
    -c,--config-file=<file>     Specify the hyper-parameter configuration file for the model [default: None]
    -e,--env=<file>             Specify the unity environment name [default: None]
    -p,--port=<n>               Specify port [default: 5005]
    -u,--unity                  Whether to use the unity client [default: False]
    -g,--graphic                Whether to display graphical interface [default: False]
    -n,--name=<name>            Specify the name of this training [default: None]
    -s,--save-frequency=<n>     Specify the frequency for saving model [default: None]
    -m,--models=<n>             How many models to train at the same time [default: 1]
    --store-dir=<file>          Specify the folder path to save the model, log, and data [default: None]
    --seed=<n>                  Specify the random seed of the model [default: 0]
    --max-step=<n>              Maximum time step per episode [default: None]
    --max-episode=<n>           Total training episodes [default: None]
    --sampler=<file>            Specify the file path for the random sampler for Unity [default: None]
    --load=<name>               Specify the name of the training to load the model [default: None]
    --fill-in                   Specify whether to pre-populate the experience pool to batch_size [default: False]
    --prefill-choose            Specify whether to choose action while pre-populate the experience pool [default: False]
    --gym                       Whether to use a gym training environment [default: False]
    --gym-agents=<n>            Specify the amount of parallel training [default: 1]
    --gym-env=<name>            Specify the name of the gym environment [default: CartPole-v0]
    --gym-env-seed=<n>          Specify random seed for gym environment [default: 0]
    --render-episode=<n>        Specify when the gym environment starts rendering [default: None]
    --info=<str>                Write a description of the training, wrapped in double quotation marks [default: None]
Example:
    python run.py --gym --gym-env Hopper-v2 -a td3 -n test --seed 0
"""

