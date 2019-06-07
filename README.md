# !!! NOTES: THIS IS AN ONGOING PROJECT !!! 

# Matrix Capsules with EM Routing
A PyTorch implementation of [Matrix Capsules with EM Routing](http://www.cs.toronto.edu/~hinton/absps/EMcapsules.pdf)

## German Traffic Sign Recognition Benchmark (GTSRB)
- Dataset [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

## Objective
The objective of this project is to use a very good reference Matrix Capsules implementation to experiment on German Traffic Sign Recognition Benchmark (GTSRB)

## Usage

1. Install [PyTorch](http://pytorch.org/) and dependencies
```bash
pip install -r requirements.txt
```

2. Start training
```bash
python train.py
```

## Test and confusion matrix

```bash
python confusion_matrix.py
```
By defaul it will load `_final_gray_model.pth` pretrained model

## Run Fast Gradient Sign Attack (Adversarial atack)
```bash
python adversarial_test.py
```

## Reference
- https://github.com/yl-1993/Matrix-Capsules-EM-PyTorch/
