#!/bin/bash

#SBATCH -J vae_train
#SBATCH -p dl
#SBATCH -o vae_train_%j.txt
#SBATCH -e vae_train_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mhaghir@iu.edu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=02:00:00

module unload python/3.6.8
export MODULEPATH=/N/soft/rhel7/modules/carbonate/DEEPLEARNING/:$MODULEPATH
module load python/3.7.3

python vae_train_self.py
