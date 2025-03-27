## Ancient Character Glyph Generation.

![image](https://github.com/CJHGray/ACGG/blob/main/picture/framework.png)

## Data preparation

### You can arbitrarily divide the training and test sets from the dataset and place them in the following format. The image names in the input folder and the target folder need to correspond one to one. The input folder stores OBS images, and the target folder stores modern Chinese character images.
```plaintext
Your_dataroot/
├── train/  (training set)
│   ├── jiagu/
│   │   ├── 1.png
│   │   ├── 2.png 
│   │   └── 3.png
│   └── jinwen/
│   │   ├── 1.png
│   │   ├── 2.png 
│   │   └── 3.png
│   │   ...
│   └── kaishu/
│       ├── 1.png
│       ├── 2.png 
│       └── 3.png
└── test/   (test set)
│   ├── jiagu/
│   │   ├── 1.png
│   │   ├── 2.png 
│   │   └── 3.png
│   └── jinwen/
│   │   ├── 1.png
│   │   ├── 2.png 
│   │   └── 3.png
│   │   ...
│   └── kaishu/
│       ├── 1.png
│       ├── 2.png 
│       └── 3.png

```

### You also need to modify the following path to configs.yaml.
```yaml
data:
    train_data_dir: '/Your_dataroot/train/' # path to directory of train data
    test_data_dir: '/Your_dataroot/test/'   # path to directory of test data
    test_save_dir: 'Your_project_path/OBS_Diffusion/result' # path to directory of test output
    val_save_dir: 'Your_project_path/OBS_Diffusion/validation/'    # path to directory of validation during training
    tensorboard: 'Your_project_path/OBS_Diffusion/logs' # path to directory of training information

training:
    resume: '/Your_save_root/diffusion_model'  # path to pretrained model
```

## Installation
```bash
conda create -n ACGG python=3.9
conda activate ACGG
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Train
```bash
python train_diffusion.py
```

### You can monitor the training process.
```bash
tensorboard --logdir ./logs
```

## Test
```bash
python eval_diffusion.py
```

