## Ancient Character Glyph Generation.

![image](https://github.com/CJHGray/ACGG/blob/main/picture/Figure_framework.pdf)

## Data preparation

### You can arbitrarily divide the training and test sets from the dataset and place them in the following format. The image names in the input folder and the target folder need to correspond one to one. The input folder stores OBS images, and the target folder stores modern Chinese character images.
```plaintext
Your_dataroot/
├── train/  (training set)
│   ├── input/
│   │   ├── train_安_1.png (OBS image)
│   │   ├── train_安_2.png 
│   │   ├── train_北_1.png
│   │   └── train_北_2.png
│   └── target/
│       ├── train_安_1.png (Modern Chinese Character image)
│       ├── train_安_2.png 
│       ├── train_北_1.png 
│       └── train_北_2.png 
│
└── test/   (test set)
    ├── input/
    │   ├── test_1.png  (OBS image)
    │   └── test_2.png
    └── target/
        ├── test_1.png  (Modern Chinese Character image)
        └── test_2.png

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

## Train

### Environment Configuration
```bash
git clone https://github.com/guanhaisu/OBSD.git
cd OBS_Diffusion
```
```bash
conda create -n OBSD python=3.9
conda activate OBSD
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```


## Start training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=1234 train_diffusion.py
```

### You can monitor the training process.
```bash
tensorboard --logdir ./logs
```

## Test
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=1234 eval_diffusion.py
```
### If you want to refine the generated character results, you can run the following script. Also be careful to change your file paths.
```bash
CUDA_VISIBLE_DEVICES=0 python refine.py
```
You can find the FontDiffuser weights here, [GoogleDrive](https://drive.google.com/drive/folders/1kRwi5sfHn6oufydDmd-7X9pPFDZzFjkk?usp=drive_link). 
