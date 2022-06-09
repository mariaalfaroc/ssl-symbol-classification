#!/bin/bash

# b-59-850

python pretrain.py --crop_scale 0.5 0.5 --sim_loss_weight 10 --var_loss_weight 1 --cov_loss_weight 1
python pretrain.py --crop_scale 0.5 1 --sim_loss_weight 10 --var_loss_weight 1 --cov_loss_weight 1
# Beam search for this corpus when using patches 
python pretrain.py --crops_labelled False --crop_scale 0.5 1.0 --sim_loss_weight 1 --var_loss_weight 1 --cov_loss_weight 1
python pretrain.py --crops_labelled False --crop_scale 0.5 1.0 --sim_loss_weight 5 --var_loss_weight 5 --cov_loss_weight 5
python pretrain.py --crops_labelled False --crop_scale 0.5 1.0 --sim_loss_weight 10 --var_loss_weight 10 --cov_loss_weight 10
python pretrain.py --crops_labelled False --crop_scale 0.5 1.0 --sim_loss_weight 1 --var_loss_weight 1 --cov_loss_weight 10
python pretrain.py --crops_labelled False --crop_scale 0.5 1.0 --sim_loss_weight 1 --var_loss_weight 10 --cov_loss_weight 1
python pretrain.py --crops_labelled False --crop_scale 0.5 1.0 --sim_loss_weight 10 --var_loss_weight 1 --cov_loss_weight 1
python pretrain.py --crops_labelled False --crop_scale 0.5 1.0 --sim_loss_weight 1 --var_loss_weight 10 --cov_loss_weight 10
python pretrain.py --crops_labelled False --crop_scale 0.5 1.0 --sim_loss_weight 10 --var_loss_weight 10 --cov_loss_weight 1
python pretrain.py --crops_labelled False --crop_scale 0.5 1.0 --sim_loss_weight 10 --var_loss_weight 1 --cov_loss_weight 10
