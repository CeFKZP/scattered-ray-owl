# Requirements

- Python 3.8.6
- Cuda 11.3
- Cuda toolkit 10.0+ (for Haste compilation)
- Pytorch 1.21.1
    - `pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113`
- Install all other requirements
    - `pip install -r requirements.txt`

# Haste Library

We use EGRU implemented in Haste library for our experiments. Here is a summary of command to compile and install haste
Refer to Haste documentation for more information.

```bash
cd haste_lib
make haste_pytorch
```

After compilation, install the wheel using pip:

```bash
pip install haste_pytorch-*.whl
```

# EGRU

Change to egru directory and add it to python path
```bash
cd ..
export PYTHONPATH=.:$PYTHONPATH
```
## Command for running sequential MNIST experiments

```bash
python bin/smnist.py --train-epochs 200 --units 590 --batch-size 500 --rnn-type egrud-haste --use-output-trace --use-grad-clipping --grad-clip-norm 0.25  --cuda
```

## Command for running the DVS Gesture experiments

```bash

python dvs_gesture/dvs128.py --data /tmp/dataset/ --cache /tmp/cache/ --logdir ./logs/ --batch-size 40 --units 795 --unit-size 1 --num-layers 1 --frame-size 128 --run-title egrud795_rerun --train-epochs 500 --frame-time 25 --rnn-type egrud-haste --learning-rate 0.0009975 --lr-gamma 0.8747 --lr-decay-epochs 56 --event-agg-method mean --use-cnn --use-all-timesteps --dropout 0.6321 --dropconnect 0.08134 --zoneout 0.2319 --pseudo-derivative-width 1.7 --threshold-mean -0.2465 --activity-regularization --activity-regularization-constant 0.01 --augment-data
```

## Language modelling experiments

Scripts and instructions can be found in the `lm` directory.
