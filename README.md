# Context Matters: Graph-based Self-supervised Representation Learning for Medical Images

Official PyTorch implementation for paper *Context Matters: Graph-based Self-supervised Representation Learning for Medical Images*, accepted by *AAAI 2021*.

Li Sun\*, Ke Yu\* and Kayhan Batmanghelich

<p align="center">
  <img width="75%" height="%75" src="./utils/model_arch.png">
</p>

### Requirements
- PyTorch 1.4
- SimpleITK
- easydict
- [torch_geometric](https://github.com/rusty1s/pytorch_geometric)
- [monai](https://github.com/Project-MONAI/MONAI)
- [tensorboard\_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
- [ANTs](http://stnava.github.io/ANTs/)

### Preprocess Data
Please follow the instructions in [./preprocess/README.md](./preprocess/README.md)

### Training
```bash
sh train.sh
```
The hyperparameter setting can be found in train.sh
### Evaluation
```
python test.py
```
