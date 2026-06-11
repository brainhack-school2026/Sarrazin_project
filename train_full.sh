#!/bin/bash
#SBATCH --account=def-jcohen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --job-name=unet_full
#SBATCH --output=/scratch/annaelle/Sarrazin_project/logs/unet_%j.out
#SBATCH --error=/scratch/annaelle/Sarrazin_project/logs/unet_%j.err

echo "Job started on $(hostname)"
echo "Time: $(date)"

module load python/3.10
source ~/brainhack/brainhack/bin/activate

cd /scratch/annaelle/Sarrazin_project

echo "Before python"

python src/Unet_train.py \
    --data_root /scratch/annaelle/Sarrazin_project \
    --manifest  training_data/manifest.csv \
    --splits    training_data/splits.json \
    --output    results/full_run_v4 \
    --epochs    20 \
    --batch_size 32

echo "Job complete at $(date)"
