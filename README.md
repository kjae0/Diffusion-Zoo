# Diffusion Zoo

<strong> Under construction... </strong> <br>
Pytorch implementation of Diffusion models.<br>
Pretrained weights are provided through GoogleDrive.

This repository aims to simplify Diffusion, easy modification.<br>


## Structure
~~~
Diffusion-Zoo
    ├── ckpts
    ├── configs
    │   ├── celebA
    │   └── cifar10
    ├── data
    │   ├── celeba
    │   └── cifar10
    ├── datasets
    │   ├── celeba_dataset.py
    │   ├── cifar10_dataset.py
    │   └── transforms.py
    ├── diffusion
    │   ├── engines
    │   ├── metrics.py
    │   ├── models
    │   ├── nn
    │   └── utils.py
    ├── eval.sh
    └── src
        ├── eval.py
        ├── __pycache__
        ├── train.py
        └── utils.py
~~~

~~~
DDPMEngine Class Structure
    ├── __init__: Initializes the model, optimizer, scheduler, datasets, and various configurations.
    ├── run_network: Runs the model with given inputs and returns predictions.
    ├── train: Manages the full training loop, saving checkpoints, and logging metrics.
    ├── train_one_epoch: Trains the model for one epoch, calculating loss and updating weights.
    ├── evaluate: Evaluates model performance using metrics like FID on test data.
    ├── postprocessing: Processes model predictions based on prediction type and clipping.
    ├── q_sample: Adds noise to input data for the diffusion process.
    ├── ddim_sample: Generates samples using DDIM sampling with optional trajectory tracking.
    ├── ddim_sample_iter: Performs a single DDIM sampling iteration.
    ├── p_sample: Generates samples using a standard diffusion sampling technique.
    ├── _p_sample_iter: Executes a single step of the sampling process.
    ├── to: Moves model and relevant components to the specified device.
    └── load_state_dict: Loads the model, optimizer, and scheduler state from a checkpoint.
~~~

## Get Started
To train model, (--enable_writer optionally)
~~~
python ./src/train.py --cfg_dir {$CONFIG_DIR} --model {$MODEL_TYPE}
~~~

To evaluate model,
~~~
python ./src/eval.py --ckpt_dir {%CKPT_DIR} --model {$MODEL_TYPE} --n_samples {$N_SAMPLES} --batch_size {$BATCH_SIZE} --dset {$DSET_TYPE}
~~~

## TODO
- [x] DDPM <br>
- [x] DDIM <br>
- [ ] CLF guid <br>
- [x] CLF free guid <br>
- [ ] NCSN <br>
- [ ] NCSN v2 <br>

## Pretrained Weights
| Model Name | Dataset | FID | IS | Weights |
|-|-|-|-|-|
| DDPM | CIFAR-10 | 2.94 | Nan | [Download](https://drive.google.com/file/d/1pVVhg2GQzUz1KWHuv1VczGD6gA9zQuI_/view?usp=drive_link) |
| DDPM | CelebA | 4.55 | Nan | [Download](https://drive.google.com/file/d/1wOw1jAY1qMEiUVBbjOcBQgm_RNJ61Rp6/view?usp=drive_link) |
| CFG | CIFAR-10 | 2.94 | Nan | [Download](https://drive.google.com/file/d/1pVVhg2GQzUz1KWHuv1VczGD6gA9zQuI_/view?usp=drive_link) |

## Examples
### <center> DDPM </center>
| CIFAR-10 | CelebA |
|-|-|
| ![990_images](https://github.com/user-attachments/assets/bc3e0e55-d259-4b0e-ad28-ae5716e1b70c) | ![290_images](https://github.com/user-attachments/assets/891207b8-7b1b-449e-9bad-37278a9bc895) |

### <center> Classifier-Free Guidance </center>
| CIFAR-10 | CelebA |
|-|-|
| ![930_images](https://github.com/user-attachments/assets/f1f0c8b6-bb53-41e7-b75a-3eeee4a71f41) |   |
COMING SOON!

