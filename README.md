# Navigating the Nuances: Comparative Analysis and Hyperparameter Optimisation of Neural Architectures on Contrast-Enhanced MRI for Liver and Liver Tumour Segmentation

This repository contains the implementation code of our
article "[Navigating the Nuances: Comparative Analysis and Hyperparameter Optimisation of Neural Architectures on Contrast-Enhanced MRI for Liver and Liver Tumour Segmentation](https://www.nature.com/articles/s41598-024-53528-9)".

It provides a comprehensive comparative analysis and hyperparameter optimization framework for seven state-of-the-art neural network models in the
field of liver and liver tumor segmentation using contrast-enhanced MRI data.

## Implemented Models

We have implemented the seven following state-of-the-art models in a single pipeline:

1. **nnFormer**: [Paper Link](https://ieeexplore.ieee.org/abstract/document/10183842)
2. **nnUNet**: [Paper Link](https://link.springer.com/chapter/10.1007/978-3-030-72087-2_11)
3. **SegmentationNet**: [Paper Link](https://ieeexplore.ieee.org/abstract/document/8363715)
4. **Swin-UNetr**: [Paper Link](https://link.springer.com/chapter/10.1007/978-3-031-08999-2_22)
5. **TransBTS**: [Paper Link](https://link.springer.com/chapter/10.1007/978-3-030-87193-2_11)
6. **UNetr**: [Paper Link](https://openaccess.thecvf.com/content/WACV2022/html/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.html)
7. **VT-UNet**: [Paper Link](https://arxiv.org/abs/2103.04430)

Despite the network implementation, this code is based on [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
and [UNetr](https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV) frameworks.

## Environment

To prepare this work we used the following setup:

- Ubuntu 20.04
- Python 3.10
- PyTorch 1.11.0
- CUDA 11.6

Once your environment set up, run the command: `pip install -r requirements.txt`

## Main Files

- `train.py`: Script to train models.
- `test.py`: Script for model inference.
- `optimisation.py`: Script to run hyperparameter optimization using Bayesian search via WandB.

## Data Preparation

The training images from Atlas dataset used in this study can be
downloaded [here](https://atlas-challenge.u-bourgogne.fr/), all the necessary information about it can be found in the associated [paper](https://www.mdpi.com/2306-5729/8/5/79).

To ensure compatibility with our processing pipeline, your dataset should be organized in the following structure:

    dataset
    │
    ├── imagesTr
    │   ├── im0.nii.gz
    │   ├── ...
    │   └── im59.nii.gz
    │
    └── labelsTr
        ├── lb0.nii.gz
        ├── ...
        └── lb59.nii.gz

Additionally, a dataset.json file is required to link training and test data. This file must contain two fields:

- training: An array of objects, each specifying the paths to an image and its corresponding label in the imagesTr and
  labelsTr directories respectively.
- test: Follows the same pattern as the training field for the test files.

The dataset.json file should look like this:

    {
        "training": [
            {
                "image": "imagesTr/im0.nii.gz",
                "label": "labelsTr/lb0.nii.gz"
            },
            ...
            {   "image": "imagesTr/im59.nii.gz",
                "label": "labelsTr/lb59.nii.gz"
            }
        ],
        "test": [
            // Follow the same pattern as training for test files
        ]
    }

## Preprocessing

To prepare the dataset you can use the `./preprocessing/preprocesing.py` script based on
the [clitk](https://github.com/benpresles/vv/wiki) library.
library. This script will automatically preprocess the dataset and generate the dataset.json file in the required format
based on the contents of your imagesTr and labelsTr directories.

To prepare the data, use `preprocessing.py`:

    python ./preprocessing/preprocessing.py --path /path/to/your/data --normalization min-max_or_range-std

If you prefer to use simple-itk use `preprocessing_sitk.py`:

    python ./preprocessing/preprocessing_sitk.py --path /path/to/your/data --normalization min-max_or_range-std 

## Configuration Files

The `config` folder contains example configuration files used to train the models mentioned in the article. The proposed
combination are the one obtained after optimisation of each model. Feel free to adapt them to your own needs.

## Training and Inference

To train a new model run the `train.py` file with an updated configuration file four your own needs:

`python train.py --config_file /path_to_your_config_file`

If you wish to perform distributed training, activate the distributed flag: `--distributed true`. The code has only been
designed to be used with the [slurm](https://slurm.schedmd.com/documentation.html) workload manager.
The pretrained weights used to
train [Swin-UNetr](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/)
and [VT-UNet](https://github.com/himashi92/VT-UNet) can be downloaded on their respective git.

To perform inference on a given model you can use the same configuration as for training, you just need to adapt the
checkpoint parameter to the location of the model you wish to use.

`python test.py --config_file /path_to_your_config_file`

## Hyperparameter tuning

To tune the hyperparameters of each model, Bayesian search based on the
WandB [sweep](https://docs.wandb.ai/guides/sweeps) function was implemented.
You can found the parameters to be found in the `config/optimisation_config.yaml` file.
To run a sweep use the following command:

`python optimisation.py --root ./config/ --model_name model_name.yaml --file optimisation_config.yaml --count number_of_model_to_be_trained_in_succession`

## To Cite

Please use the following BibTeX entry to cite our work:

```bibtex
@article{quinton2024navigating,
  title={Navigating the nuances: comparative analysis and hyperparameter optimisation of neural architectures on contrast-enhanced MRI for liver and liver tumour segmentation},
  author={Quinton, Felix and Presles, Benoit and Leclerc, Sarah and Nodari, Guillaume and Lopez, Olivier and Chevallier, Olivier and Pellegrinelli, Julie and Vrigneaud, Jean-Marc and Popoff, Romain and Meriaudeau, Fabrice and others},
  journal={Scientific Reports},
  volume={14},
  number={1},
  pages={3522},
  year={2024},
  publisher={Nature Publishing Group UK London}
}