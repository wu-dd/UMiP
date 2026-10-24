# UMiP

Official PyTorch implementation of  
**Making Partial-Label Datasets Easier: A Simple Yet Highly Effective Data Augmentation for Deep Partial-Label Learning**  
(ECCV 2026). [[paper]](https://openreview.net/pdf?id=swEVq3Ycr3)

The repository plugs UMiP into six PLL algorithms: PRODEN (ICML 2020),
LWS (ICML 2021), CRDPLL (ICML 2022), PiCO (ICLR 2022),
ABLE (IJCAI 2022) and DIRK (AAAI 2024). `--mode orig` runs a base algorithm
unchanged, `--mode UMiP` adds UMiP to it. The UMiP branch of each algorithm
is the block guarded by `if self.mode == "UMiP":` in `core/algorithms.py`.

## Setup

```bash
conda create -n umip python=3.9
conda activate umip
pip install -r requirements.txt
```

## Data

CIFAR-10, CIFAR-100, SVHN, Fashion-MNIST and KMNIST are downloaded
automatically by torchvision into `./data/<name>/`. For CUB-200, place
`CUB_200_2011.tgz` under `./data/CUB200/`.

Candidate label sets are generated from the clean training labels with
`--pl_type` and `--pl_degree`:

- `proden_uniform`: every false label is added independently with probability `pl_degree`.
- `class_dependent`: labels are flipped to similar classes with probability `pl_degree` (10-class datasets).
- `instance_dependent`: labels are added according to the predictions of a clean classifier, scaled by `pl_degree`. This needs the generator checkpoints from [Google Drive](https://drive.google.com/drive/folders/1odvsu7QGJIPh2MEXtAyBSfkdN5E7dXSg?usp=sharing), placed in `data/partial_models_pool/partial_weights/` (see the README there).
- `uniform`, `aggregate`, `vaguest`, `sup`: classic uniform sets, human-annotated PLCIFAR10 sets (needs `./data/plcifar10/plcifar10.pkl`), and a supervised sanity check.

## Running

```bash
# PRODEN + UMiP on CIFAR-10 with instance-dependent candidate labels
python train.py --dataset CIFAR10 --model ResNet32 --pl_type instance_dependent --pl_degree 0.8 \
  --algorithm PRODEN --mode UMiP --seed 1

# the corresponding baseline
python train.py --dataset CIFAR10 --model ResNet32 --pl_type instance_dependent --pl_degree 0.8 \
  --algorithm PRODEN --mode orig --seed 1
```

## Citation

```bibtex
@inproceedings{wu2026making,
  title={Making Partial-Label Datasets Easier: A Simple Yet Highly Effective Data Augmentation for Deep Partial-Label Learning},
  author={Wu, Dong-Dong and Li, Zhaoyi and Li, Xiang and Shen, Zhiqiang},
  booktitle={European Conference on Computer Vision},
  pages={506--524},
  year={2026},
  organization={Springer}
}
```
