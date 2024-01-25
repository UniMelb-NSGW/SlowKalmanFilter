#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=48:00:00 
#SBATCH --job-name=canonical_1_seed_1250 
#SBATCH --output=outputs/canonical_1_seed_1250_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py configs/canonical_1_seed_1250.ini