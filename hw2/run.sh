#!/usr/bin/env bash
#python train_pg_f18.py --exp_name="vpg_rtg_na_s1"  -rtg --seed 1
#python train_pg_f18.py --exp_name="vpg_rtg_na_s1_1"  -rtg --seed 1

python train_pg_f18.py --env_name InvertedPendulum-v2 --exp_name vpg_rtg_dna_lr0.01_s1 -rtg -dna --seed 1 -lr 0.01
python train_pg_f18.py --env_name InvertedPendulum-v2 --exp_name vpg_rtg_dna_lr0.01_s1_1 -rtg -dna --seed 1 -lr 0.01