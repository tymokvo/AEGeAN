"""
Deeper DCGAN with AE stabilization.

Use/distribute freely but plz attribute.

I hope you have a good GPU.
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from AEGeAN_1024 import GAN_D, GAN_G
from torch.autograd import Variable
from torch.backends import cudnn
import time as t
cudnn.benchmark = True


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


to_image = transforms.ToPILImage()

date = t.strftime('%y-%m-%d')
time = t.strftime('%H-%M-%S')

# these settings push a gtx1080 to the limit...
nf = 8
n_iter = 1000
z_d = 100
batch_size = 4
image_size = 1024
workers = 4
num_checkpoints = 7

print('will save every {} epochs'.format(n_iter // num_checkpoints))

dataroot =  # INSERT DATA DIRECTORY HERE
# Useful for making many models
base_dir = '.../modelsAEGeAN1024_{}_{}/'.format(dataroot.split('/')[-1], date)
checkpoint_path = base_dir + 'models/'
save_dir = base_dir + 'images/'

dirs = [base_dir, checkpoint_path, save_dir]
for D in dirs:
    if not os.path.exists(D):
        os.mkdir(D)

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Scale(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=int(workers))


# ---------------
# Nets
# ---------------

_Encoder = GAN_D(3, nf, nout=z_d)
_Decoder = GAN_G(z_d, nf, nc=3)
_Discriminator = GAN_D(3, nf, nout=1)

# Replace final activation to match DCGAN
_Discriminator.layers['act_9'] = nn.Sigmoid()

_Encoder.build()
_Decoder.build()
_Discriminator.build()

_Encoder.apply(init_weights)
_Decoder.apply(init_weights)
_Discriminator.apply(init_weights)

load_models = False
if load_models:
    _Encoder.load_state_dict(torch.load())
    _Decoder.load_state_dict(torch.load())
    _Discriminator.load_state_dict(torch.load())

_Encoder.cuda()
_Decoder.cuda()
_Discriminator.cuda()

nets = [_Encoder, _Decoder, _Discriminator]

[print(net.n_params()[0]) for net in nets]

# ---------------
# Variables
# ---------------

noise = torch.FloatTensor(batch_size, z_d, 1, 1)

# Tried using a uniform distribution to allow the model to maintain more points in latent space. Didn't really work.
fixed_noise = torch.FloatTensor(batch_size, z_d, 1, 1).uniform_(-10, 10)
torch.save(fixed_noise, base_dir + 'fixed_noise.pth')

noise = Variable(noise).cuda()
fixed_noise = Variable(fixed_noise).cuda()

# ---------------
# Loss/Optimizers
# ---------------

criterion = nn.BCELoss(size_average=True).cuda()

optim_encoder = optim.SGD(_Encoder.parameters(), lr=1e-3, momentum=0.9)
optim_decoder = optim.SGD(_Decoder.parameters(), lr=1e-3, momentum=0.9)
optim_decoderGAN = optim.Adam(_Decoder.parameters(), lr=2e-4)
optim_discriminator = optim.SGD(
    _Discriminator.parameters(), lr=2e-4, momentum=0.9)

# these can be noised/annealed if needed
real_label = 1.0
fake_label = 0.0

# add explicit epochs to save
save_list = [30, 60, 90]

n_image_rows = int(math.sqrt(batch_size))

for epoch in range(n_iter):
    for i, (batch, _) in enumerate(dataloader):
        [net.zero_grad() for net in nets]

        current_batch_size = batch.size(0)

        # ---------------
        # Train as AE
        # ---------------

        input = Variable(batch).cuda()

        encoded = _Encoder(input)
        encoded = encoded.unsqueeze(0)
        encoded = encoded.view(current_batch_size, -1, 1, 1)
        reconstructed = _Decoder(encoded, mode='ae')

        reconstruction_loss = criterion(reconstructed, input)
        reconstruction_loss.backward()

        optim_encoder.step()
        optim_decoder.step()
        _Decoder.zero_grad()

        # ---------------
        # Train as GAN
        # ---------------

        # Train Discriminator on real

        # _Discriminator.zero_grad()
        real_samples = input.clone()
        inference_real = _Discriminator(real_samples)
        labels = torch.FloatTensor(current_batch_size).fill_(real_label)
        labels = Variable(labels).cuda()
        real_loss = criterion(inference_real, labels)
        real_loss.backward()

        # Generate fake samples

        noise.data.resize_(current_batch_size, z_d, 1, 1)
        noise.data.uniform_(-10, 10)
        #noise.data.normal_(0, 1)
        fake_samples = _Decoder(noise, mode='gen')

        # Train Discriminator on fake

        labels.data.fill_(fake_label)
        inference_fake = _Discriminator(fake_samples.detach())
        fake_loss = criterion(inference_fake, labels)
        fake_loss.backward()
        discriminator_total_loss = real_loss + fake_loss
        optim_discriminator.step()

        # Update Decoder/Generator with how well it fooled Discriminator

        # _Decoder.zero_grad()
        labels.data.fill_(real_label)
        inference_fake_Decoder = _Discriminator(fake_samples)
        fake_samples_loss = criterion(inference_fake_Decoder, labels)
        fake_samples_loss.backward()
        optim_decoderGAN.step()

        print('[{:4d}/{:4d}][{:4d}/{:4d}] Loss_Re: {:.4f} Loss_D: {:.4f} Loss_G: {:.4f}'.format(
            epoch + 1, n_iter, i + 1, len(dataloader), reconstruction_loss.data[0], discriminator_total_loss.data[0], fake_samples_loss.data[0])
        )
        if i % 35 == 0:
            real_grid = vutils.make_grid(
                batch, n_image_rows, 0, normalize=True)
            real_grid = to_image(real_grid)
            real_grid.save('{}/real_samples.gif'.format(save_dir))

            decoded_grid = vutils.make_grid(
                reconstructed.data.clone().cpu(), n_image_rows, 0, normalize=True)
            decoded_grid = to_image(decoded_grid)
            decoded_grid.save(
                '{}/decoded_samples_epoch_{}_{}.gif'.format(save_dir, epoch, i))

            fake = _Decoder(fixed_noise, mode='gen')
            fake_grid = vutils.make_grid(
                fake.data.clone().cpu(), n_image_rows, 0, normalize=True)
            fake_grid = to_image(fake_grid)
            fake_grid.save(
                '{}/fake_samples_epoch_{}_{}.gif'.format(save_dir, epoch, i))

    # save n checkpoints (with optional list)
    if epoch % (n_iter // num_checkpoints) == 0 and epoch is not 0 or epoch in save_list:
        torch.save(_Encoder.state_dict(), '{}Encoder_nf{}_e{}_recon-err{:.4f}.pth'.format(
            checkpoint_path, nf, epoch, reconstruction_loss.data[0]))
        torch.save(_Decoder.state_dict(), '{}Decoder_nf{}_e{}_fake-err{:.4f}.pth'.format(
            checkpoint_path, nf, epoch, fake_samples_loss.data[0]))
        torch.save(_Discriminator.state_dict(), '{}Discriminator_nf{}_e{}_discrim-err{:.4f}.pth'.format(
            checkpoint_path, nf, epoch, discriminator_total_loss.data[0]))

# Final save just in case!
torch.save(_Encoder.state_dict(), '{}Encoder_nf{}_e{}_recon-err{:.4f}.pth'.format(
    checkpoint_path, nf, epoch, reconstruction_loss.data[0]))
torch.save(_Decoder.state_dict(), '{}Decoder_nf{}_e{}_fake-err{:.4f}.pth'.format(
    checkpoint_path, nf, epoch, fake_samples_loss.data[0]))
torch.save(_Discriminator.state_dict(), '{}Discriminator_nf{}_e{}_discrim-err{:.4f}.pth'.format(
    checkpoint_path, nf, epoch, discriminator_total_loss.data[0]))
