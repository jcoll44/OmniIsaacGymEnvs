# Agent Metacognition

## About this repository

This repository contains a metacognition RL environment. It includes code for trainging an agent to complete the taska and code for collecting the training and test datasets. 

Once the datasets are collected, metacognitive agents can be trained and assessed using the metacognitive_models repository found here.........

## Training an agent

Start the docker container

```bash
./docker/run_docker.sh
```

Either train your own agent

```bash
/isaac-sim/python.sh scripts/rlgames_train.py headless=True task=Jackal wandb_activate=True wandb_group=Jackal wandb_entity=jcoll44
```

OR test the pretrained agent and view it using the [Omniverse Streaming Client](https://docs.omniverse.nvidia.com/app_streaming-client/app_streaming-client/overview.html).

```bash
/isaac-sim/python.sh scripts/rlgames_train.py task=Jackal checkpoint=runs/Jackal/nn/Jackal.pth test=True headless=True enable_livestream=True
```

## Collecting the training and test datasets

Run the following script to collect the training and test datasets for the Jackal task. The datasets will be saved in the `data` folder.

```bash
/isaac-sim/python.sh scripts/rlgames_meta_dataset.py task=Jackal headless=True test=True checkpoint=runs/Jackal/nn/Jackal.pth enable_livestream=False num_envs=1
```
