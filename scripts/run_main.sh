#!/usr/bin/env bash
# Main benchmark sweep: every base PLL algorithm, with and without UMiP,
# on CIFAR-10 / CIFAR-100 / SVHN / CUB-200 under the three candidate-label
# generation protocols used in the paper. Run from the repository root.

for mode in "orig" "UMiP"
do
  for seed in 1 2 3
  do
    for ds in "CIFAR10" "CIFAR100" "SVHN" "CUB200"
    do
      model="ResNet32"
      [[ "$ds" == "CUB200" ]] && model="ResNet34"

      for pl_type in "proden_uniform" "instance_dependent" "class_dependent"
      do
        # class-dependent generation is defined for 10-class datasets only
        [[ "$pl_type" == "class_dependent" && ( "$ds" == "CIFAR100" || "$ds" == "CUB200" ) ]] && continue

        pl_degree=(0.0)
        [[ "$pl_type" == "proden_uniform" && ( "$ds" == "CIFAR10" || "$ds" == "SVHN" ) ]] && pl_degree=(0.3 0.5 0.7)
        [[ "$pl_type" == "proden_uniform" && "$ds" == "CIFAR100" ]] && pl_degree=(0.01 0.05 0.1)
        [[ "$pl_type" == "proden_uniform" && "$ds" == "CUB200" ]] && pl_degree=(0.01 0.02 0.03)

        [[ "$pl_type" == "instance_dependent" && ( "$ds" == "CIFAR10" || "$ds" == "SVHN" ) ]] && pl_degree=(0.6 0.8 1.0)
        [[ "$pl_type" == "instance_dependent" && "$ds" == "CIFAR100" ]] && pl_degree=(0.01 0.05 0.1)
        [[ "$pl_type" == "instance_dependent" && "$ds" == "CUB200" ]] && pl_degree=(0.01 0.02 0.03)

        [[ "$pl_type" == "class_dependent" && "$ds" == "CIFAR10" ]] && pl_degree=(0.7 0.8 0.9)
        [[ "$pl_type" == "class_dependent" && "$ds" == "SVHN" ]] && pl_degree=(0.7 0.8 0.9)

        for degree in "${pl_degree[@]}"
        do
          for algo in "PRODEN" "LWS" "CRDPLL" "PiCO" "DIRK" "ABLE"
          do
            mkdir -p log
            python train.py --dataset ${ds} --model ${model} --pl_type ${pl_type} --pl_degree ${degree} \
              --algorithm ${algo} --seed ${seed} --mode ${mode} \
              > ./log/${mode}_${algo}_${ds}_${model}_${pl_type}_${degree}_${seed}.log 2>&1
          done
        done
      done
    done
  done
done
