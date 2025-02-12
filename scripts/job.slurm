#!/bin/bash
#SBATCH --job-name=negotiation_rl
#SBATCH --nodes=1  
#SBATCH --ntasks=1       
#SBATCH --mem=32G  
#SBATCH --gres=gpu:1   
#SBATCH --time=24:00:00  
#SBATCH --mail-type=begin       
#SBATCH --mail-type=end         
#SBATCH --mail-user=zs7353@princeton.edu
module purge
module load anaconda3/2024.2
conda activate LLMNegotiation
sh /home/zs7353/LLMNegotiation/scripts/run.sh