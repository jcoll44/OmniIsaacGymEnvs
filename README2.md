# Agent Metacognition

## About this repository

This repository contains a metacognition RL environment. It includes collecting the training script for training an agent, a script for collecting the training and test datasets. 

Once the datasets are collected, metacognitive agents can be trained and assessed using the repository found here.........

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
PYTHON_PATH scripts/rlgames_confidence_dataset_collection.py task=Jackal headless=True checkpoint=runs/Jackal/nn/Jackal.pth
```
