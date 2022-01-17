import argparse
import os
import numpy as np
import pandas as pd
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm, trange

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

torch.cuda.empty_cache

epoche = 3000

os.makedirs("images", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=epoche, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=96, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=2, help="number of training steps for discriminator per iter")
parser.add_argument("--n_save", type=int, default=epoche//5, help="number of epoch to save the model")
parser.add_argument("--sample_interval", type=int, default=500, help="interval betwen image samples")
parser.add_argument("--n_old_state", type=int, default=4, help="number of model in memory")
parser.add_argument('--generator_input', nargs='+', type=int, default=[256,256,512,1024,1024,2048]) #[310,310,610,1220,1220,2440]
parser.add_argument('--discriminator_input', nargs='+', type=int)
parser.add_argument("--folder_model", default="data/", help="path of models save")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
iDis = 0
iGen = 0
learning_rate = opt.lr

cuda = True if torch.cuda.is_available() else False



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.75))
            layers.append(nn.Dropout(p=0.25))
            layers.append(nn.LeakyReLU(0.25, inplace=True))
            return layers

        arr = [*block(opt.latent_dim, opt.generator_input[0], normalize=False)]
        for i in range(0,len(opt.generator_input)-1):
            arr.extend([*block(opt.generator_input[i],opt.generator_input[i+1])])
        arr.append(nn.Linear(opt.generator_input[-1], int(np.prod(img_shape))))
        arr.append(nn.Tanh())
        self.model = nn.Sequential(*arr)

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
# Loss weight for gradient penalty
lambda_gp = 2

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()


firstD = True
firstG = True

# ------------
#  Load Model
# ------------
if not os.path.exists(opt.folder_model):
    os.mkdir(opt.folder_model)
else:
    print("####Load: Generator and Discriminator")
    

for i in range(opt.n_old_state,-1,-1):
    if os.path.exists(opt.folder_model+"gen_"+str(i)+".c002") and iGen==0:
        if firstG: 
            firstG= False
        else:
            print("    Generator: ",end="")
            generator = (torch.load(opt.folder_model+"gen_"+str(i)+".c002"))
            print("loaded")
            iGen = i
            
    if os.path.exists(opt.folder_model+"dis_"+str(i)+".c002") and iDis==0:
        if firstD: 
            firstD= False
        else:
            print("    Discriminator: ",end="")
            discriminator =(torch.load(opt.folder_model+"dis_"+str(i)+".c002"))
            print("loaded")
            iDis = i
            
        


if cuda:
    generator.cuda()
    discriminator.cuda()
    
    
def arrToNp(x):
    temp = []
    image = torch.zeros([0])
    for i,im in enumerate(x,0):
      image = torch.Tensor(np.array(im.split(",")).reshape(3,96,96).astype('double')/255)
      temp.append(image)
    return temp
    
print("####Load: database")
print("    Read: ",end="")
df = pd.read_csv('data/file3.csv', sep=',', index_col=None)
ice = [int(i) for i in (np.random.rand(1,300)*630).reshape(300)]
print("OK",end="")
elements = [item for sublist in [[x+650*y for y in range(4)] for x in range(155)] for item in sublist]
x = df["Male"][elements]
print("\n    Prepare data(%d): "%(len(x)),end="")
dataframe = arrToNp(x)


    
# Configure data loader
dataloader = torch.utils.data.DataLoader(
    dataframe,
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True
)
print("OK")
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    if len(d_interpolates.shape)==4 : d_interpolates = d_interpolates[:,:,-1,-1]
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# ----------
#  Training
# ----------

batches_done = 0
dloss=""
gloss=""
srtr = ""

for epoch in range(opt.n_epochs):
    for i, (imgs) in enumerate(dataloader):
        srtr=""
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z)
        
        

        # Real images
        real_validity = discriminator(real_imgs)
        
        # Fake images
        fake_validity = discriminator(fake_imgs)
        
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        
        
        d_loss.backward()

        optimizer_D.step()
        
        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
			
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -(torch.mean(fake_validity))

            g_loss.backward()
            optimizer_G.step()
            dloss = d_loss.item()
            gloss = g_loss.item()
            srtr = "".join(["1" if abs(x.item())>0.5 else "0" for x in fake_validity])
            print(
                "[Epochs {:12s}] [Batch {:7s}] [D loss: {:6s}] [G loss: {:6s}] {:s}"
                    .format(str(epoch)+"/"+str(opt.n_epochs), str(i)+"/"+str(len(dataloader)), str(dloss)[0:8], str(gloss)[0:8],srtr)
            )

            if batches_done % opt.sample_interval == 0:
                save_image(fake_imgs.data[:opt.batch_size], "images/%d.png" % batches_done, nrow=math.floor(math.sqrt(opt.batch_size)), normalize=True)

            batches_done += opt.n_critic
    
	# ------------
	#  Save Model
	# ------------
	
    if epoch%opt.n_save==0 and epoch!=0:
        if iDis>0 and iGen>0:
            if os.path.exists(opt.folder_model+"dis_0.c002"):
                os.remove(opt.folder_model+"dis_0.c002")
            if os.path.exists(opt.folder_model+"gen_0.c002"):
                os.remove(opt.folder_model+"gen_0.c002")
        if iDis==opt.n_old_state: 
            for i in range(1,iDis+1):
                if os.path.exists(opt.folder_model+"dis_"+str(i)+".c002"):
                    os.rename(opt.folder_model+"dis_"+str(i)+".c002",opt.folder_model+"dis_"+str(i-1)+".c002")
        if iGen==opt.n_old_state: 
            for i in range(1,iGen+1):
                if os.path.exists(opt.folder_model+"gen_"+str(i)+".c002"):
                    os.rename(opt.folder_model+"gen_"+str(i)+".c002",opt.folder_model+"gen_"+str(i-1)+".c002")
                
        torch.save(generator, opt.folder_model+"gen_"+str(iGen)+".c002")
        torch.save(discriminator, opt.folder_model+"dis_"+str(iDis)+".c002")
        
        iDis = min((iDis+1), opt.n_old_state)
        iGen = min((iGen+1), opt.n_old_state)
    
print("####Saving: ",end="")
if iDis>0 and iGen>0:
    if os.path.exists(opt.folder_model+"dis_0.c002"):
        os.remove(opt.folder_model+"dis_0.c002")
    if os.path.exists(opt.folder_model+"gen_0.c002"):
        os.remove(opt.folder_model+"gen_0.c002")
    if iDis==opt.n_old_state: 
        for i in range(1,iDis+1):
            print(".",end="")
            if os.path.exists(opt.folder_model+"dis_"+str(i)+".c002"):
                os.rename(opt.folder_model+"dis_"+str(i)+".c002",opt.folder_model+"dis_"+str(i-1)+".c002")
            if os.path.exists(opt.folder_model+"gen_"+str(i)+".c002"):
                os.rename(opt.folder_model+"gen_"+str(i)+".c002",opt.folder_model+"gen_"+str(i-1)+".c002")
    print(".",end="")
    torch.save(discriminator, opt.folder_model+"dis_"+str(iDis)+".c002")
    print(".",end="")
    torch.save(generator, opt.folder_model+"gen_"+str(iGen)+".c002")
    print(" OK",end="")
    iDis = min((iDis+1), opt.n_old_state)
    iGen = min((iGen+1), opt.n_old_state)