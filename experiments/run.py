import sys

sys.path.append("./")

from experiments.config import DS_PRETRAIN_HPARAMS, DS_TEST_HPARAMS
from pretrain import run_pretrain as pretrain
from test_knn import run_bootstrap as test_knn
from train import run_bootstrap as train


def run_approach_experiments(ds_pretrain_hparams: dict, ds_test_hparams: dict):
    for ds_name, ds_pretrain_config in ds_pretrain_hparams.items():
        ############################ PROPOSED APPROACH
        # 1) First, pretrain
        # VICReg pretrained using labeled bounding boxes
        # and random patches (Custom CNN and Resnet34)
        for model_type in ["CustomCNN", "Resnet34"]:
            pretrain(
                ds_name=ds_name,
                supervised_data=True,
                model_type=model_type,
                **ds_pretrain_config,
            )
            pretrain(
                ds_name=ds_name,
                supervised_data=False,
                model_type=model_type,
                **ds_pretrain_config,
            )

        # 2) Then, test using KNN
        for spc in [1, 5, 10, 15, 20, 25, 30]:
            for model_type in ["CustomCNN", "Resnet34"]:
                # VICReg pretrained using labeled bounding boxes
                # and random patches (Custom CNN and Resnet34)
                test_knn(
                    ds_name=ds_name,
                    samples_per_class=spc,
                    model_type=model_type,
                    pretrained=True,
                    checkpoint_path=ds_test_hparams[ds_name][
                        f"{model_type.lower()}_bboxes"
                    ],
                )
                test_knn(
                    ds_name=ds_name,
                    samples_per_class=spc,
                    model_type=model_type,
                    pretrained=True,
                    checkpoint_path=ds_test_hparams[ds_name][
                        f"{model_type.lower()}_patches"
                    ],
                )

            ############################ END PROPOSED APPROACH

            ############################ BASELINE APPROACH
            # Flatten and pretrained Resnet34 on ImageNet
            test_knn(
                ds_name=ds_name,
                samples_per_class=spc,
                model_type="Flatten",
            )
            test_knn(
                ds_name=ds_name,
                samples_per_class=spc,
                model_type="Resnet34",
                pretrained=True,
            )
            # Supervised learning (Custom CNN and Resnet34)
            for model_type in ["CustomCNN", "Resnet34"]:
                train(
                    ds_name=ds_name,
                    samples_per_class=spc,
                    model_type=model_type,
                    pretrained=False,
                    epochs=ds_pretrain_config["epochs"],
                    batch_size=ds_pretrain_config["batch_size"],
                )
            ############################ END BASELINE APPROACH


if __name__ == "__main__":
    run_approach_experiments(DS_PRETRAIN_HPARAMS, DS_TEST_HPARAMS)
