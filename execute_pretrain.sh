#!/bin/bash

##################### b-59-850 #####################
# ---- Pretrain:
for base_model in CustomCNN Resnet34 Vgg19; do
    # Crops labelled
    python -u pretrain.py --ds_path b-59-850 --base_model $base_model --crop_scale 0.5 1.0 --sim_loss_weight 10.0 --var_loss_weight 10.0 --num_randomcrops ALL > b-59-850_CropsLabelled_$base_model\_ALL.logs
    for num_randomcrops in ALL 5000 10000 15000 20000; do
        python -u pretrain.py --ds_path b-59-850 --base_model $base_model --crops_labelled False --crop_scale 0.5 1.0 --sim_loss_weight 10.0 --var_loss_weight 10.0 --num_randomcrops $num_randomcrops > b-59-850_CropsUnlabelled_$base_model\_$num_randomcrops.logs
    done
done
# ---- Test:
for base_model in CustomCNN Resnet34 Vgg19; do
    python -u test.py --ds_path b-59-850 --base_model $base_model --model_name TODO --weights_path TODO --pretrain_conv TrueTODO


    parser.add_argument("--model_name", type=str, default=None, help="Model name", required=True)
    parser.add_argument("--weights_path", type=str, default=None, help="Weights path to load")
    


    parser.add_argument("--pretrain_conv", type=str2bool, default="True", help="Pretrain convolutional models")


