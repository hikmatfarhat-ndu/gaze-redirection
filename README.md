# Implementation of gaze redirection in PyTorch

A PyTorch implementation of this [paper](https://arxiv.org/abs/1903.12530). It was implemented in Tensorflow1 [here](https://github.com/HzDmS/gaze_redirection)

### Training
 
Use `train.py` for training where the parameters are taken from `config.yaml`.

Another version of `train.py`, `train-fabric.py`, uses lightning Fabric with `bf16-mixed` precision and accomplishes about 25% speedup on NVIDIA RTX3070.

### Testing

The `transform.py` transforms the gaze of an input image. The generator file and the input are required. The gaze is given as `--angles H V` where H and V are the horizontal and vertical angles respectively. If no angles are specified the V is set to 0 and H in the set [-15,-10,-5,0,5,10,15] 