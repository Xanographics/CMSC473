#!/bin/bash
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=16:00:00

python3 version2.py

