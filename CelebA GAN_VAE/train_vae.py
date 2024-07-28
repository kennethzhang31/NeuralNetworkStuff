import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
from read_celeba import *

# Decide which device we want to run on
# if torch.backends.mps.is_available():
device = torch.device("mps")
# else:
#     device = torch.device("cpu")

# Construct Model
z_dim = 128
netEnc = models.Encoder(z_dim).to(device)
netDec = models.Generator(z_dim).to(device)
params = list(netEnc.parameters()) + list(netDec.parameters())
opt = optim.Adam(params, lr=2e-4, betas=(0.5, 0.999))

# Create Results Folder
model_name = "vae"
out_folder = "out/" + model_name + "/"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
save_folder =  "save/" + model_name + "/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Main Loop
num_epochs = 20
z_fixed = torch.randn(36, z_dim, device=device)
print("Start Training ...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # Initialize
        netEnc.zero_grad()
        netDec.zero_grad()

        # Forward 
        '''
        TODO: Forward Computation of the Network
        1. Compute z_mean and z_logvar.
        2. Sample z from z_mean and z_logvar.
        3. Compute reconstruction of x.
        '''
        real_data = data[0].to(device)
        z_mean, z_logvar = netEnc(real_data)
        std = torch.exp(z_logvar * 0.5)
        eps = torch.randn_like(std)
        z = z_mean + eps * std
        z.to(device)
        reconstructed_x = netDec(z)
        # Loss and Optimization
        '''
        TODO: Loss Computation
        rec_loss = 
        kl_loss = 
        '''
        rec_loss = F.mse_loss(reconstructed_x, real_data)
        kl_loss = 0.5 * torch.mean(torch.sum( -z_logvar + torch.exp(z_logvar) - 1 + torch.pow(z_mean, 2), dim=1))

        loss = rec_loss + 0.0001*kl_loss
        loss.backward()
        opt.step()

        # Show Information
        if i % 100 == 0:
            print("[%d/%d][%s/%d] R_loss: %.4f | KL_loss: %.4f"\
            %(epoch+1, num_epochs, str(i).zfill(4), len(dataloader), rec_loss.item(), kl_loss.mean().item()))
        
        if i % 500 == 0:
            print("Generate Images & Save Models ...")
            # Output Images
            x_fixed = netDec(z_fixed).cpu().detach()
            plt.figure(figsize=(6,6))
            plt.imshow(np.transpose(vutils.make_grid(x_fixed, nrow=6, padding=2, normalize=True).cpu(),(1,2,0)))
            plt.axis("off")
            plt.savefig(out_folder+str(epoch).zfill(2)+"_"+str(i).zfill(4)+".jpg", bbox_inches="tight")
            plt.close()
            # Save Model
            torch.save(netEnc.state_dict(), save_folder+"netEnc.pt")
            torch.save(netDec.state_dict(), save_folder+"netDec.pt")
            