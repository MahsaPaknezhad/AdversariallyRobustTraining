# Gradient Regularization

This repository contains the code for the paper entitled "Explaining Adversarial Vulnerability with a Data Sparsity Hypothesis".

# Dependency Installation

The following dependencies are required to run the code in this repository:

- `python 3.6`
- `numpy 1.19.1`
- `pytorch 1.3.1` 
- `tqdm 4.47.0`
- `torchvision 0.4.2` 
- `matplotlib 3.2.2`
- `scipy 1.5.0`
- `pillow 7.2.0`
- `tensorboard 2.2.1`
- `scikit-image 0.16.2`
- `foolbox 3.0.4`
- `torch-optimizer 0.0.1a14`

# Features

The main features of this repository are:

1. `main.py`: Master file to train models. (adversarial training, gradient regularization, etc.)
2. `shellfiles`: Directory including pre-configured shell files for convenience of set-up, training, and analysis of environment and models.
3. `TensorBoard support`: Library allowing for real-time analysis and easy downloading of model training curves.
4. `getAcc.py`: File to obtain robust accuracy of trained models. Adversarial attacks can be configured.
5. `getRho.py`: File to obtain rho values of trained models. Adversarial attacks can be configured.

# Usage Walkthrough

- After `downloading` or `git clon`ing this repository, run `prepare_environment.sh` under `shellfiles`, to prepare the `reg_env` conda environment.

- To train `MNIST, CIFAR10, or Imagenette` models, run the corresponding `train_template.sh` under `shellfiles`, adjusting any parameters as necessary.
  - While these files are running, you can set up `TensorBoard` to monitor progress with the procedure described below:
    1. Run `tensorboard --log_path <folder>`, where `<folder>` is the ancestor of all directories containing model files and results.
        - e.g. `../output/CIFAR10_Results/CIFAR10s` when in this folder.
    2. Using your browser of choice, connect to `http://localhost:6006`.
- After the models' training complete, run the corresponding `acc_template.sh` or `rho_template.sh` to obtain the <img src="https://render.githubusercontent.com/render/math?math=\hat{\rho}_{adv}" width="4%" height="3%"> and robust accuracy values.
- To obtain test accuracy values, run the corresponding `testacc_template.sh`.

# Imagenette Dataset

The link is taken from the fastai imagenette repo: see https://github.com/fastai/imagenette <br>
Use the following link to download the dataset: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz