#!/bin/bash
#SBATCH --job-name=XLSR_FT_A
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurms/XLSR_FT_A-%j.out
#SBATCH --error=slurms/XLSR_FT_A-%j.err

source ~/.bashrc
conda activate dep_model
cd ~/projects/Depression_Detection/Model\ training\ code

# 確保已安裝 opensmile
pip install opensmile --quiet

python xlsr_egemaps_dann_finetune_A.py
