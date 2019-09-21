import torchvision.models as models
import torch 
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class VggDataset(Dataset):
    def __init__(self, sr_path, gt_path):
        self.sr_path, self.gt_path = sr_path, gt_path
        self.sr_names = os.listdir(sr_path)
        self.gt_names = os.listdir(gt_path)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def preprocess(self, img_path):
        temp_img = Image.open(img_path)
        return self.transform(temp_img)

    def __getitem__(self, index):
        sr_img = self.preprocess(os.path.join(self.sr_path, self.sr_names[index]))
        gt_img = self.preprocess(os.path.join(self.gt_path, self.gt_names[index]))

        return sr_img.to(device), gt_img.to(device)
        
    def __len__(self):
        return len(self.sr_names)

vgg19 = models.vgg19(pretrained=True).to(device)
vgg19.eval()

vgg_data = VggDataset(sr_path='../results/RRDB_ESRGAN_x4/div-2k-test', gt_path='../data_samples/div_train/valid_HR')
vgg_loader = DataLoader(vgg_data, batch_size=2)

for sr, gt in vgg_loader:
    sr_vgg = vgg19(sr)
    gt_vgg = vgg19(gt)

    vgg_loss = ((gt_vgg - sr_vgg) ** 2).mean()
    print(vgg_loss)
