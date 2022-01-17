# LinearGAN
A simple GAN built with Linear layer

INSTRUCTION


the network accepts images as array of int or float whit shape like this:

  B x N x W x H
  
B : Number of batch<br />
N : the number of channel (RGB=3)<br />
W : the image width<br />
H : the height image<br />

python:<br />
usage: gan.py [-h] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--b1 B1] [--b2 B2] [--n_cpu N_CPU] [--latent_dim LATENT_DIM] [--img_size IMG_SIZE] [--channels CHANNELS] [--n_critic N_CRITIC] [--n_save N_SAVE] [--sample_interval SAMPLE_INTERVAL] [--n_old_state N_OLD_STATE] [--generator_input GENERATOR_INPUT [GENERATOR_INPUT ...]] [--folder_model FOLDER_MODEL]<br />

optional arguments:<br />
  -h, --help &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; show this help message and exit<br />
  --n_epochs N_EPOCHS&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp; number of epochs of training<br />
  --batch_size BATCH_SIZE&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;size of the batches<br />
  --lr LR&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;adam: learning rate<br />
  --b1 B1&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&ensp;&nbsp;adam: decay of first order momentum of gradient<br />
  --b2 B2&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&ensp;&nbsp;adam: decay of first order momentum of gradient<br />
  --n_cpu N_CPU&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&ensp;number of cpu threads to use during batch generation<br />
  --latent_dim LATENT_DIM&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;dimensionality of the latent space<br />
  --img_size IMG_SIZE&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;size of each image dimension<br />
  --channels CHANNELS&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;number of image channels<br />
  --n_critic N_CRITIC&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;number of training steps for discriminator per iter<br />
  --n_save N_SAVE&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&ensp;&nbsp;number of epoch to save the model<br />
  --sample_interval SAMPLE_INTERVAL&ensp;&emsp;interval betwen image samples<br />
  --n_old_state N_OLD_STATE&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;&emsp;number of model in memory<br />
  --generator_input GENERATOR_INPUT [GENERATOR_INPUT ...]<br />
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;generator layers size<br />
  --folder_model FOLDER_MODEL&emsp;&emsp;&emsp;&nbsp;path of models save<br />
