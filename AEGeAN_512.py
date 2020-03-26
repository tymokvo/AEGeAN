import torch
from torch.autograd import Variable
import torch.nn as nn
from collections import OrderedDict as odict


class GAN_G(nn.Module):
    def __init__(self, nz, ngf, nc=3):
        super(GAN_G, self).__init__()
        self.main = nn.Sequential()
        self.layers = odict([
            # block 0
            ('convT_0', nn.ConvTranspose2d(nz, ngf * \
                                           32, 4, stride=2, padding=1, bias=False)),
            ('bn_0', nn.BatchNorm2d(ngf * 32)),
            ('act_0', nn.LeakyReLU(inplace=True)),
            # block 1
            ('convT_1', nn.ConvTranspose2d(ngf * 32, ngf * \
                                           16, 4, stride=2, padding=1, bias=False)),
            ('bn_1', nn.BatchNorm2d(ngf * 16)),
            ('act_1', nn.LeakyReLU(inplace=True)),
            # block 2
            ('convT_2', nn.ConvTranspose2d(ngf * 16, ngf * \
                                           16, 4, stride=2, padding=1, bias=False)),
            ('bn_2', nn.BatchNorm2d(ngf * 16)),
            ('act_2', nn.LeakyReLU(inplace=True)),
            # block 3
            ('convT_3', nn.ConvTranspose2d(ngf * 16, ngf * \
                                           16, 4, stride=2, padding=1, bias=False)),
            ('bn_3', nn.BatchNorm2d(ngf * 16)),
            ('act_3', nn.LeakyReLU(inplace=True)),
            # block 4
            ('convT_4', nn.ConvTranspose2d(ngf * 16, ngf * \
                                           32, 4, stride=2, padding=1, bias=False)),
            ('bn_4', nn.BatchNorm2d(ngf * 32)),
            ('act_4', nn.LeakyReLU(inplace=True)),
            # block 5
            ('convT_5', nn.ConvTranspose2d(ngf * 32, ngf * \
                                           16, 4, stride=2, padding=1, bias=False)),
            ('bn_5', nn.BatchNorm2d(ngf * 16)),
            ('act_5', nn.LeakyReLU(inplace=True)),
            # block 6
            ('convT_6', nn.ConvTranspose2d(ngf * 16, ngf * \
                                           16, 4, stride=2, padding=1, bias=False)),
            ('bn_6', nn.BatchNorm2d(ngf * 16)),
            ('act_6', nn.LeakyReLU(inplace=True)),
            # block 7
            ('convT_7', nn.ConvTranspose2d(ngf * 16,
                                           ngf * 8, 4, stride=2, padding=1, bias=False)),
            ('bn_7', nn.BatchNorm2d(ngf * 8)),
            ('act_7', nn.LeakyReLU(inplace=True)),
            # block 8
            ('convT_8', nn.ConvTranspose2d(
                ngf * 8, nc, 4, stride=2, padding=1, bias=False)),
            ('act_8', nn.LeakyReLU(inplace=True)),
        ])
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def build(self):
        for name, layer in self.layers.items():
            self.main.add_module(name, layer)

    def forward(self, x, mode='gen'):
        out = self.main(x)
        if mode == 'gen':
            out = self.tanh(out)
        if mode == 'ae':
            out = self.sigmoid(out)
        return out

    def n_params(self):
        params = [[num for num in param.size()]
                  for param in self.main.parameters()]

        n = 0
        param_graph = []
        for group in params:
            s = 1
            for item in group:
                s *= item
            param_graph += [s]
            n += s
        return n, param_graph


class GAN_D(nn.Module):
    def __init__(self, nc, ndf, nout):
        super(GAN_D, self).__init__()
        self.nout = nout
        self.main = nn.Sequential()
        self.layers = odict([
            # block 0
            # input: (batch x nc x 512 x 512)
            ('conv_1', nn.Conv2d(nc, ndf * 8, 4, stride=2, padding=1, bias=False)),
            ('bn_1', nn.BatchNorm2d(ndf * 8)),
            ('act_1', nn.LeakyReLU(0.2, inplace=True)),
            # block 2
            # input: (batch x nc x 256 x 256)
            ('conv_2', nn.Conv2d(ndf * 8, ndf * 16,
                                 4, stride=2, padding=1, bias=False)),
            ('bn_2', nn.BatchNorm2d(ndf * 16)),
            ('act_2', nn.LeakyReLU(0.2, inplace=True)),
            # block 3
            # input: (batch x nc x 128 x 128)
            ('conv_3', nn.Conv2d(ndf * 16, ndf * 16,
                                 4, stride=2, padding=1, bias=False)),
            ('bn_3', nn.BatchNorm2d(ndf * 16)),
            ('act_3', nn.LeakyReLU(0.2, inplace=True)),
            # block 4
            # input: (batch x nc x 64 x 64)
            ('conv_4', nn.Conv2d(ndf * 16, ndf * 16,
                                 4, stride=2, padding=1, bias=False)),
            ('bn_4', nn.BatchNorm2d(ndf * 16)),
            ('act_4', nn.LeakyReLU(0.2, inplace=True)),
            # block 5
            # input: (batch x nc x 32 x 32)
            ('conv_5', nn.Conv2d(ndf * 16, ndf * 16,
                                 4, stride=2, padding=1, bias=False)),
            ('bn_5', nn.BatchNorm2d(ndf * 16)),
            ('act_5', nn.LeakyReLU(0.2, inplace=True)),
            # block 6
            # input: (batch x nc x 16 x 16)
            ('conv_6', nn.Conv2d(ndf * 16, ndf * 32,
                                 4, stride=2, padding=1, bias=False)),
            ('bn_6', nn.BatchNorm2d(ndf * 32)),
            ('act_6', nn.LeakyReLU(0.2, inplace=True)),
            # block 7
            # input: (batch x nc x 8 x 8)
            ('conv_7', nn.Conv2d(ndf * 32, ndf * 32,
                                 4, stride=2, padding=1, bias=False)),
            ('bn_7', nn.BatchNorm2d(ndf * 32)),
            ('act_7', nn.LeakyReLU(0.2, inplace=True)),
            # block 8
            # input: (batch x nc x 4 x 4)
            ('conv_8', nn.Conv2d(ndf * 32, ndf * 32,
                                 4, stride=2, padding=1, bias=False)),
            ('bn_8', nn.BatchNorm2d(ndf * 32)),
            ('act_8', nn.LeakyReLU()),
            # block9
            # input: (batch x nc x 2 x 2)
            ('conv_9', nn.Conv2d(ndf * 32, nout, 2, stride=2, padding=0, bias=False)),
            ('act_9', nn.LeakyReLU()),
        ])

    def build(self):
        for name, layer in self.layers.items():
            self.main.add_module(name, layer)

    def forward(self, x):
        out = self.main(x)
        return out.view(-1, self.nout)

    def n_params(self):
        params = [[num for num in param.size()]
                  for param in self.main.parameters()]

        n = 0
        param_graph = []
        for group in params:
            s = 1
            for item in group:
                s *= item
            param_graph += [s]
            n += s
        return n, param_graph


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = torch.randn(1, 100, 1, 1)
    net = GAN_G(100, 24, 3)
    net.build()

    np, pg = net.n_params()

    print('{:,}'.format(np))
    # net.cuda()

    out = net(x)
    print(out.size())

    xs = range(len(pg))
    plt.bar(xs, pg, color='k')
    plt.yscale('log')
    for xloc, v in zip(xs, pg):
        plt.text(xloc, max(pg) / 2,
                 s='{:,}'.format(v), color=(1, 0, 1), rotation=90)
    plt.show()
