#!/bin/bash
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
source ~/virtualenv/serena_sleeping/bin/activate


python ~/ANN_ESM2_3B_O_linked_glycosylation_Independent_Testing.py