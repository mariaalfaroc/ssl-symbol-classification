#!/bin/bash

# This is an example of how to run the experiments
# appearing in the paper for one of the considered datasets

##################### b-59-850 #####################
# ---- Pretrain:
# NOTE
# We eventually run this experiment with 5 iterations
# but the following code only considers 1 iteration
for base_model in CustomCNN Resnet34 Vgg19; do
    # Crops labelled
    python -u pretrain.py --ds_path b-59-850 --base_model $base_model --crop_scale 0.5 1.0 --sim_loss_weight 10.0 --var_loss_weight 10.0 --num_randomcrops ALL > b-59-850_$base_model\_LabelledCrops-ALL.logs
    for num_randomcrops in ALL 5000 10000 15000 20000; do
        python -u pretrain.py --ds_path b-59-850 --base_model $base_model --crops_labelled False --crop_scale 0.5 1.0 --sim_loss_weight 10.0 --var_loss_weight 10.0 --num_randomcrops $num_randomcrops > b-59-850_$base_model\_UnlabelledCrops-$num_randomcrops.logs
    done
done
# ---- Test:
# Previously trained models
for base_model in CustomCNN Resnet34 Vgg19; do
    python -u test.py --ds_path b-59-850 --base_model $base_model --weights_path b-59-850/experiments/Labelled_True_kernel_\(64\,\ 64\)_stride_\(32\,\ 32\)_delLines_False_MOD_$base_model\_ENC_1600_EXP_1024_s_10.0_v_10.0_c1.0_crop_True_scale_\(0.5\,\ 1.0\)_numCrops_ALL.pt > b-59-850_test_Model$base_model\_LabelledCrops-ALL.logs
    for num_randomcrops in ALL 5000 10000 15000 20000; do
        python -u test.py --ds_path b-59-850 --base_model $base_model --weights_path b-59-850/experiments/Labelled_False_kernel_\(64\,\ 64\)_stride_\(32\,\ 32\)_delLines_False_MOD_$base_model\_ENC_1600_EXP_1024_s_10.0_v_10.0_c1.0_crop_True_scale_\(0.5\,\ 1.0\)_numCrops_$num_randomcrops.pt > b-59-850_test_Model$base_model\_UnlabelledCrops-$num_randomcrops.logs
    done
done
# Flatten
python -u test.py --ds_path b-59-850 --model_name Flatten --n_iterations 5 > b-59-850_Flatten.logs
# Supervised with CustomCNN
python -u test.py --ds_path b-59-850 --model_name Supervised --n_iterations 5 > b-59-850_CustomCNNSupervised.logs
# Supervised with Resnet34 and Vgg19 (ResnetClassifier and VggClassifier); Transfer Learning with Resnet34 and Vgg19(ResnetEncoder and VggEncoder)
for model_name in ResnetEncoder ResnetClassifier VggEncoder VggClassifier; do
    for pretrained in True False; do
        python -u test.py --ds_path b-59-850 --model_name $model_name --n_iterations 5 --pretrain_conv $pretrained > b-59-850_$model_name\_Pretrained-$pretrained.logs
    done
done
