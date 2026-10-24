# Partial-label generator checkpoints

`--pl_type instance_dependent` builds candidate label sets from the softmax
output of a classifier trained with clean labels on the same training set.
The checkpoints are not tracked in git. Download them from

https://drive.google.com/drive/folders/1odvsu7QGJIPh2MEXtAyBSfkdN5E7dXSg?usp=sharing

and place them in this directory:

- `cifar10.pt`: WideResNet-28-10 (`wide_resnet.py`)
- `cifar100.pt`: torchvision `wide_resnet50_2` (DataParallel)
- `svhn.pt`: WideResNet-28-10 (`wide_resnet.py`)
- `cub200.pt`: torchvision `wide_resnet50_2` (DataParallel)
- `fmnist.pt`: MLP (`mlp.py`)
- `kmnist.pt`: MLP (`mlp.py`)

Other `--pl_type` values (`proden_uniform`, `uniform`, `class_dependent`)
do not need any checkpoint.
