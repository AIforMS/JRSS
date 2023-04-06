#!/bin/bash
#SBATCH -J mono-reg
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o logs/trainOut.txt
#SBATCH -e logs/trainErr.txt

SLURM_SUBMIT_DIR=~/JRSS
cd $SLURM_SUBMIT_DIR
NP=$SLURM_JOB_GPUS

CUDA_VISIBLE_DEVICES=$NP python train_mono_reg.py \
    -backbone swinunet \
    -dataset ct-mr \
    -batch_size 1 \
    -interval 1 \
    -lr 1e-4 \
    -weakly_sup \
    -apply_ti_loss \
    -use_checkpoint \
    -apply_lr_scheduler \
    -feature_size 24 \
    -epochs 1500 \
    -alpha 4.0 \
    -sim_weight 1.0 \
    -seg_weight 1.0 \
    -ti_weight 1e-5 \
    -sim_loss MIND \
    -output output/swinunet_mono/ \
    -mov_folder path/to/mov_folder \
    -mov_name ?_bcv_CT.nii.gz \
    -fix_folder path/to/fix_folder \
    -fix_name ?_chaos_MR.nii.gz \
    -train_pair_numbers "1 2 3 4 6 7 8 9" \
    -val_pair_numbers "5 10"
