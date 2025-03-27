## Ancient Character Glyph Generation.

![image](https://github.com/CJHGray/ACGG/blob/main/picture/framework.png)

## Data preparation
We present the structure of our dataset, and you can create your own dataset in the same format.
```plain-text
font-607/
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

## Evaluate the generated results
Modify this part of the metrics.py to compare the generated characters with the real character images.
```bash
generated_img_dir = './ACGG-result/result'
real_img_dir = './font-607/test/zhuanshu'
```
```bash
python metrics.py
```

