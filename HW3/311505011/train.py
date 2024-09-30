

# In[1]:


import torch.nn.functional as F
import torch
import torchvision
import matplotlib.pyplot as plt
from pytorch_gan_metrics import get_fid
from torchvision.io import read_image
import glob
import os
from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
import math
from torch.optim import Adam
from tensorboardX import SummaryWriter
import argparse
import functions
from tqdm import tqdm
import models
from torchvision.utils import save_image


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64).float()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).float()

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                    produces the cumulative product of (1-beta) up to that
                    part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                    prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--T', type=int, default=500)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--beta_scheduler', type=str, default="linear")
    global args
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    IMG_SIZE = args.img_size
    

    

    # Define beta schedule
    T = args.T
    

    if args.beta_scheduler == 'linear':
        betas = linear_beta_schedule(timesteps=T)
    elif args.beta_scheduler == 'cosine':
        betas = cosine_beta_schedule(timesteps=T)
    elif args.beta_scheduler == 'cosine_improved':
        betas = get_named_beta_schedule("cosine", T)
        betas = torch.tensor(betas).float()
    print("beta schedular: ", args.beta_scheduler)

    # Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


    name = f"{args.name}_batch{BATCH_SIZE}_lr{args.lr}_T{args.T}"
    device = torch.device('cuda:'+args.device)

    logdir = "./runs/"
    os.makedirs(logdir,exist_ok=True)
    train_writer = SummaryWriter(os.path.join(logdir)+'/'+name)

    name = name + '/'

    imagedir = "./generated_images/"+name
    os.makedirs(imagedir,exist_ok=True)


    
    weights_dir = './weights/'+name
    os.makedirs(weights_dir,exist_ok=False)
    check_point_path = weights_dir + 'ckpts/'
    os.makedirs(check_point_path,exist_ok=False)
    save_path = weights_dir+'best.pth'


    # In[2]:



    train_dir = './mnist'
    data_list = sorted(glob.glob(os.path.join(train_dir,'*png')))
    print('Number of dataset:',len(data_list))


    # In[4]:



    data = functions.load_transformed_dataset(IMG_SIZE)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,num_workers=16)


    
    #model = functions.SimpleUnet().to(device)
    model = models.Unet(dim = 64,dim_mults = (1,2,2)).to(device)
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    #gaussian_diffusion = models.GaussianDiffusion(timesteps=T)


    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=10,verbose=1)



    epochs = args.epochs # Try more!
    len_train_data = len(dataloader)
    best_FID = float("Inf")
    FID = 0
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        
        with tqdm(total=len_train_data, ncols=100, position=0, leave=True, desc="Training: ") as pbar:
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()

                t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
                #print(t.shape)
                #print(batch[0].shape)
                loss_b = functions.get_loss(model, batch, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
                #loss_b = gaussian_diffusion.train_losses(model, batch, t)
                loss_b.backward()
                running_loss += loss_b
                optimizer.step()
                current_lr=optimizer.param_groups[0]['lr']

                pbar.set_postfix_str(f"loss_b: {loss_b:.4f}")
                pbar.update(1)
        loss = running_loss/float(len_train_data)
        scheduler.step(loss)
        train_writer.add_scalar(tag='loss', scalar_value=float(loss), global_step=epoch)
        train_writer.add_scalar(tag='lr', scalar_value=float(current_lr), global_step=epoch)
        
        

        
        model.eval()
        best = ''
        with torch.no_grad():
            FID, display_images = functions.test(model,device, IMG_SIZE, T, BATCH_SIZE, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)
            print("FID: ",FID)
            torch.save(model.state_dict(), check_point_path+f'ckpt_{epoch}.pt')
            if FID <= best_FID:
                best_FID = FID
                torch.save(model.state_dict(), save_path)
                print("Best weights saved!!")
                best = 'best'
                save_image(display_images, imagedir+f'epoch{epoch}_{best}_FID{FID}.png')
            #print(display_images.shape)
            
            """
            for idx, img in enumerate(generated_images):
                if idx%100==0:
                    functions.save_tensor_image(img, imagedir+f'_{epoch}_{idx}.png')
            """
        train_writer.add_scalar(tag='FID', scalar_value=float(FID), global_step=epoch)
        
            

        print(", ".join([
                f"Epoch {epoch:3d}/{args.epochs:3d}",
                f"train_loss: {loss:.4f}",
                f"lr: {current_lr}", 
                f"FID: {FID}"
            ]))
        
if __name__ == '__main__':
    main()



