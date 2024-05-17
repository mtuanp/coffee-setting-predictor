# coffee-grinder-predictor

This project is a small educational project for predicting my espresso setting by deep learning. It uses a simple feedforward neural network with two hidden layers.

## My espresso setup

- Lelit Mara X V2 | PL62X 
- TIMEMORE Electric Coffee Grinder Sculptor 78S
- TIMEMORE Basic 2.0 Electronic Espresso Scale with Timer
- Coffee from my local coffee roaster

## Goal of the prediction

1. predict the optimal coffee grinder setting for a 1:2 ratio, grinded coffee to extracted coffee
2. predict the optimal extraction time for that ratio

## Requirements

- Python 3.12.3

## Initial setup

1. ``python3 -m venv ./venv``
2. ``source ./venv/bin/activate``
3. ``python3 -m pip install -r requirements.txt``

## How to use it

1. ``source ./venv/bin/activate``
2. ``python3 -m grinder_model_trainer``
3. ``python3 -m grinder_predictor``

## Plans

1. web ui for input the test data 
2. web ui for calling the predictor