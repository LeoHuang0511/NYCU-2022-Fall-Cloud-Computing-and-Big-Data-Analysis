import torch.nn.functional as F
import torch
import torchvision
import matplotlib.pyplot as plt
#from pytorch_gan_metrics import get_fid, get_inception_fea
from metrics import get_fid
from torchvision.io import read_image
from torch.utils.data import Dataset
import os
from PIL import Image
import PIL
from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
import math
import glob
from tqdm import tqdm
from torchvision.utils import save_image
import functions
import models
import argparse

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64).float()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--T', type=int, default=500)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--beta_scheduler', type=str, default="linear")
    parser.add_argument('--weights', type=str, default="./weights/_batch64_lr0.0005_T500/best.pth")
    global args
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    IMG_SIZE = args.img_size
    

    

    # Define beta schedule
    T = args.T
    
    weights_path = args.weights
   
    betas = linear_beta_schedule(timesteps=T)
    
    print("beta schedular: ", args.beta_scheduler)

    # Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


    #name = f"{args.name}_batch{BATCH_SIZE}_lr{args.lr}_T{args.T}"
    device = torch.device('cuda:'+args.device)
    model = models.Unet(dim = 64,dim_mults = (1,2,2)).to(device)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model.load_state_dict(torch.load(weights_path,map_location='cuda:'+args.device))
    

    

    
    
    FID, display_images, generated_images = generate_images(model, device, IMG_SIZE, T, BATCH_SIZE, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas,posterior_variance)

    weights_name = os.path.split(os.path.split(weights_path)[0])[1].strip('.pth')
    imagedir = f"./final_generated_images/{weights_name}/"
    os.makedirs(imagedir,exist_ok=False)
    images_path = imagedir + 'images/'
    os.makedirs(images_path,exist_ok=True)
    difusion_path = imagedir + 'diffusion_process/'
    os.makedirs(difusion_path,exist_ok=True)

    save_image(display_images, difusion_path+'diffusion_process.png')
    for idx, imgs in enumerate(generated_images):
        save_image(imgs, images_path+f'{idx+1:05d}.png')




@torch.no_grad()
def generate_images(model, device, IMG_SIZE, T, BATCH_SIZE, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas,posterior_variance):
    # Sample noise
    img_size = IMG_SIZE
    
    #plt.figure(figsize=(15,15))
    #plt.axis('off')
    num_images = 10000
    t = torch.full((1,), 0, device=device, dtype=torch.long)
    imgs = torch.randn((num_images, 3, img_size, img_size))
    testset = functions.TestDataset(imgs)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE*3, shuffle=False, drop_last=False,num_workers=16)
    len_test_data = len(testloader)
    #stepsize = int(T/num_images)
    model.eval()
    with torch.no_grad():
        generated_imgs = torch.tensor([]).to(device)
        num_display = 8
        diffusion_step = 7
        time_step = int(T/diffusion_step)
        display_images = torch.tensor([]).to(device)
        with tqdm(total=len_test_data, ncols=100, position=0, leave=True, desc="Generating: ") as pbar:
            for idx, batch in enumerate(testloader):
                batch = batch.to(device)
                #print(batch.device)
                for i in range(0,T)[::-1]:
                    t = torch.full((1,), i, device=device, dtype=torch.long)
                    #print(model(img,t).shape)
                    batch = functions.sample_timestep(model, batch, t, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas,posterior_variance)
                    if idx==0 and i%time_step==0:
                        display_images = torch.cat((display_images,batch[:num_display]))
                        #print(batch[:num_display].shape)
                
                generated_imgs = torch.cat((generated_imgs,batch))
                pbar.update(1)
        
        FID = get_fid(generated_imgs, './mnist.npz',device=device)
    return FID, display_images, generated_imgs
                
    
if __name__ == '__main__':
    main()
