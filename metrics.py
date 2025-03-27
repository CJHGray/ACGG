import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from piq import psnr, ssim
from torch.nn.functional import cosine_similarity
from torch.xpu import device
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import lpips
import cv2
import numpy as np
import torchvision.models as models

# 定义图像预处理变换
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: binarize_image(x))
    # transforms.Lambda(lambda x: x.type(torch.uint8))
])

def binarize_image(img_tensor):
    img_np = img_tensor.numpy().transpose(1, 2, 0) * 255
    img_np = img_np.astype(np.uint8)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, binary_img = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    binarize_image_tensor = torch.from_numpy(binary_img).unsqueeze(0).float()/255.0
    return binarize_image_tensor

model = models.resnet18(pretrained=True)
new_conv1 = torch.nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
new_conv1.weight.data = model.conv1.weight.data.mean(dim=1,keepdim=True)
model.conv1 = new_conv1
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 初始化评估指标
psnr_value = 0
ssim_value = 0
lpips_metric = lpips.LPIPS(net='alex').to('cuda' if torch.cuda.is_available() else 'cpu')
lpips_value = 0
cosine_similarity_value = 0

# generated_img_dir = './OBS_Diffusion/result'
generated_img_dir = './ACGG-result/result'

real_img_dir = './font-607/test/zhuanshu'
# real_img_dir = './OBS_Diffusion/result'


# 获取生成图片和真实图片的文件名列表
generated_image_files = sorted([f for f in os.listdir(generated_img_dir) if f.endswith('.png')])
real_image_files = sorted([f for f in os.listdir(real_img_dir) if f.endswith('.png')])

# 确保生成图片和真实图片数量一致
assert len(generated_image_files) == len(real_image_files), "生成图片和真实图片数量不一致"

# 遍历每张图片进行计算
for generated_file, real_file in zip(generated_image_files, real_image_files):
    # 构建生成图片和真实图片的路径
    generated_image_path = os.path.join(generated_img_dir, generated_file)
    # print(generated_image_path)
    real_image_path = os.path.join(real_img_dir, real_file)

    # 加载并预处理生成图片
    generated_image = Image.open(generated_image_path).convert('RGB')
    generated_image = transform(generated_image)
    generated_image = generated_image.unsqueeze(0)

    # 加载并预处理真实图片
    real_image = Image.open(real_image_path).convert('RGB')
    real_image = transform(real_image)
    real_image = real_image.unsqueeze(0)

    # 将图像数据移动到相应设备上
    generated_image = generated_image.to('cuda' if torch.cuda.is_available() else 'cpu')
    real_image = real_image.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 计算PSNR和SSIM
    psnr_value += psnr(generated_image, real_image).item()
    ssim_value += ssim(generated_image, real_image).item()

    lpips_value += lpips_metric(generated_image, real_image).item()

    with torch.no_grad():
        generated_features = model(generated_image).view(1, -1)
        real_features = model(real_image).view(1, -1)
    cosine_similarity_value += F.cosine_similarity(generated_features, real_features).item()

# 计算平均PSNR和SSIM
num_images = len(generated_image_files)
average_psnr = psnr_value / num_images
average_ssim = ssim_value / num_images

average_lpips = lpips_value / num_images

average_cosine_similarity = cosine_similarity_value / num_images

print(f"平均PSNR值: {average_psnr}")
print(f"平均SSIM值: {average_ssim}")
print(f"平均LPIPS值: {average_lpips}")
print(f"平均cos: {average_cosine_similarity}")