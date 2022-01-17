# LinearGAN
A simple GAN built with Linear layer

INSTRUCTION


the network accepts images as array of int or float whit shape like this:

  B x N x W x H
  
B : Number of batch
N : the number of channel (RGB=3)
W : the image width
H : the height image

python
usage: gan.py [-h] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--b1 B1] [--b2 B2] [--n_cpu N_CPU] [--latent_dim LATENT_DIM] [--img_size IMG_SIZE] [--channels CHANNELS] [--n_critic N_CRITIC] [--n_save N_SAVE] [--sample_interval SAMPLE_INTERVAL] [--n_old_state N_OLD_STATE] [--generator_input GENERATOR_INPUT [GENERATOR_INPUT ...]] [--folder_model FOLDER_MODEL]

optional arguments:
  -h, --help                          show this help message and exit
  --n_epochs N_EPOCHS                 number of epochs of training
  --batch_size BATCH_SIZE             size of the batches
  --lr LR                             adam: learning rate
  --b1 B1                             adam: decay of first order momentum of gradient
  --b2 B2                             adam: decay of first order momentum of gradient
  --n_cpu N_CPU                       number of cpu threads to use during batch generation
  --latent_dim LATENT_DIM             dimensionality of the latent space
  --img_size IMG_SIZE                 size of each image dimension
  --channels CHANNELS                 number of image channels
  --n_critic N_CRITIC                 number of training steps for discriminator per iter
  --n_save N_SAVE                     number of epoch to save the model
  --sample_interval SAMPLE_INTERVAL   interval betwen image samples
  --n_old_state N_OLD_STATE           number of model in memory
  --generator_input GENERATOR_INPUT [GENERATOR_INPUT ...]
                                      generator layers size
  --folder_model FOLDER_MODEL         path of models save
