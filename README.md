
# Open-set Label Noise Can Improve Robustness Against Inherent Label Noise

NeurIPS 2021: 
This repository is the official implementation of [ODNL](https://arxiv.org/abs/2106.10891). 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py cifar10 --alg odnl -m wrn --noise_type symmetric --noise_rate 0.4 --exp_name test --gpu 0 --lambda_o 3.0
```


## Evaluation

To evaluate the model on CIFAR-10, run:

```eval
python test.py cifar10 --method_name cifar10_symmetric_04_wrn_test_odnl --num_to_avg 10 --gpu 0 --seed 1 --prefetch 0 --out_as_pos
```

## Hyperparameter

The best test accuracy (%) and the value of \eta on CIFAR-10/100 using vanilla ODNL is shown as follow:

| Dataset | Method | Sym-20% | Sym-50% | Asym | Dependent | Open |
| :-----| ----: | :----: | :----: | :----: | :----: | :----: |
| CIFAR-10 | Ours | 91.06  | 82.50  | 90.00  | 85.37 | 91.47   |
| - |  \eta | 2.5 | 2.5  | 3.0   | 3.5  | 2.0 |
| CIFAR-100 | Ours | 68.82  | 54.08  | 58.61  | 62.45 | 66.95   |
| - |  \eta | 1.0 | 1.0  | 2.0   | 2.0  | 1.0 |

## Datasets

You can download ***300K Random Images*** datasets (from [OE](https://github.com/hendrycks/outlier-exposure)) in the following url:

### [300K Random Images](https://people.eecs.berkeley.edu/~hendrycks/300K_random_images.npy)


## What's More?
Below are my other research works related to this topic:

1. Can we use OOD examples to rebalance long-tailed dataset? [ICML 22](https://arxiv.org/pdf/2206.08802.pdf) | [Code](https://github.com/hongxin001/open-sampling)
2. How to handle noisy labels in domain adaptation: [AAAI 2022](https://arxiv.org/pdf/2201.06001.pdf) | [Code](https://github.com/Renchunzi-Xie/GearNet)
3. How to handle multiple noisy labels? [TNNLS](https://hongxin001.github.io/docs/papers/2022TNNLS.pdf)
4. Combating noisy labels with Agreement: [CVPR 2020](https://arxiv.org/pdf/2003.02752.pdf) | [Code](https://github.com/hongxin001/JoCoR)

<!-- 1. Using open-set noisy labels to improve robustness against inherent noisy labels: [NeurIPS 2021](https://arxiv.org/pdf/2106.10891.pdf) | [Code](https://github.com/hongxin001/ODNL) -->


## Citation

If you find this useful in your research, please consider citing:

    @article{wei2021odnl,
      title={Open-set Label Noise Can Improve Robustness Against Inherent Label Noise},
      author={Wei, Hongxin and Tao, Lue and Xie, Renchunzi and An, Bo},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2021}
    }

