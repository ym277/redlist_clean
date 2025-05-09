#!/bin/bash
#SBATCH -J dev
#SBATCH --output=dev.out
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=64
#SBATCH -p full
#SBATCH --gres=gpu:1
 
echo "starting vis at `date` on `hostname`"

# activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dam

job_name=$1


which python


echo "python train.py ${job_name}"
python train.py ${job_name}