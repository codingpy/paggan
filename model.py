import copy

import glob

import collections

import wget

# git clone https://github.com/NVIDIA/apex
# cd apex
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

import apex

import caffe

import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import cv2

import numpy as np


class PAGGAN:
    '''Learning Face Age Progression: A Pyramid Architecture of GANs

    `Improved Techniques for Training GANs` checklist:
        [ ] feature matching
        [ ] minibatch discrimination
        [ ] historical averaging
        [v] one-sided label smoothing
        [ ] virtual batch normalization

    `Connecting Generative Adversarial Networks and Actor-Critic Methods` checklist:
        [v] replay buffers
        [ ] target networks
        [ ] entropy regularization
        [ ] compatibility
    '''

    def __init__(self):
        self.netD = IntegratedDiscriminator()
        self.netG = Generator()
        self.netE = DeepFaceDescriptor()

    def train(self, x1, x2):
        torch.backends.cudnn.benchmark = True

        self.netD.cuda()
        self.netG.cuda()
        self.netE.cuda()

        for param in self.netD.net1.parameters():
            param.requires_grad = False

        for param in self.netE.parameters():
            param.requires_grad = False

        adam1_wd = \
            torch.optim.Adam(self.netD.parameters(), lr=1e-4, weight_decay=0.5)
        adam2_wd = \
            torch.optim.Adam(self.netG.parameters(), lr=1e-4, weight_decay=0.5)

        adam1 = torch.optim.Adam(self.netD.parameters(), lr=1e-4)
        adam2 = torch.optim.Adam(self.netG.parameters(), lr=1e-4)

        loader1 = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(x1)),
            batch_size=8,
            shuffle=True,
            drop_last=True,
        )
        loader2 = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(x2)),
            batch_size=8,
            shuffle=True,
            drop_last=True,
        )

        data1 = foreveriter(loader1)
        data2 = foreveriter(loader2)

        buffer = ReplayBuffer(size=50)

        # penalizes the samples depending on how close they are to
        # the decision boundary in a metric space, minimizing the
        # Pearson X^2 divergence
        criterion = nn.MSELoss()

        real = torch.tensor(1.).cuda()
        fake = torch.tensor(0.).cuda()

        for i in range(50000):
            # uses Adam with the learning rate of 1 x 10^-4 and the
            # weight decay factor of 0.5 for every 2,000 iterations
            if i % 2000 == 0:
                optimizerD = adam1_wd
                optimizerG = adam2_wd
            else:
                optimizerD = adam1
                optimizerG = adam2

            x1_real = next(data1)[0].float().cuda()
            x2_real = next(data2)[0].float().cuda()

            # updates the discriminator at every iteration

            optimizerD.zero_grad()

            y1_real = self.netD(x1_real)
            y2_real = self.netD(x2_real)

            x2_fake = self.netG(x1_real)

            y2_fake = self.netD(buffer(x2_fake.detach()))

            # feeds both the actual young faces and the generated
            # age-progressed faces into D as negative samples while
            # the true elderly images as positive ones

            loss = (
                criterion(y1_real, fake)
                + criterion(y2_real, real - .25)
                + criterion(y2_fake, fake)
            ) / 3
            loss.backward()

            optimizerD.step()

            # uses the age-related and identity-related critics
            # at every generator iteration

            optimizerG.zero_grad()

            y2_fake = self.netD(x2_fake)

            h1_real = self.netE(x1_real)
            h2_fake = self.netE(x2_fake)

            # sets the trade-off parameters to 0.20, 750.00 and
            # 0.005 for CACD

            loss = 750 * criterion(y2_fake, real) \
                + .005 * criterion(h1_real, h2_fake)

            # employs the pixel-level critic for every 5 generator
            # iterations
            if i % 5 == 0:
                loss += .2 * criterion(x1_real, x2_fake)

            loss.backward()

            optimizerG.step()

            if i % 100 == 0:
                # BGR -> RGB
                x1_real = x1_real.flip(1)
                x2_fake = x2_fake.flip(1)

                torchvision.utils.save_image(
                    [*x1_real, *x2_fake], f'{i:05}.png', normalize=True)


class IntegratedDiscriminator(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # age estimation
        self.net1 = torchvision.models.vgg16()

        # removes the fully connected layers
        self.net1.classifier = nn.Identity()

        if pretrained:
            params = self.net1.parameters()

            # 3 x 224 x 224
            wget.download('https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt')
            wget.download('https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel')

            # docker pull bvlc/caffe:gpu
            dex = caffe.Net('age.prototxt', caffe.TEST, weights='dex_chalearn_iccv2015.caffemodel')

            tensors = []

            for param in dex.params.values():
                for i in range(2):
                    tensors.append(torch.Tensor(param[i].data))

            for param, tensor in zip(params, tensors):
                param.data.copy_(tensor)

            # python/caffe/imagenet/ilsvrc_2012_mean.npy
            mean = torch.Tensor(
                # BGR
                [[[104.00698793]], [[116.66876762]], [[122.67891434]]]
            )

            self.register_buffer('mu', mean)

        self.net2 = Discriminator()

    def forward(self, x):
        # subtracts the image mean from each image
        h = x - self.mu

        x = []

        i = 0

        for module in self.net1.features.children():
            h = module(h)

            # uses the 2nd, 4th, 7th and 10th convolutional layers
            if isinstance(module, nn.Conv2d):
                if i in [1, 3, 6, 9]:
                    x.append(h)

                i += 1

        # jointly estimates the pyramid facial feature representations
        y = self.net2(x)
        return y


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # supplementary material architecture

        self.net1 = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, 4, stride=2, padding=1),
        )

        self.net2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            copy.deepcopy(self.net1)
        )

        self.net3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            copy.deepcopy(self.net2)
        )

        self.net4 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            copy.deepcopy(self.net3)
        )

    def forward(self, x):
        # pathways
        x4, x3, x2, x1 = x

        x1 = self.net1(x1)
        x2 = self.net2(x2)
        x3 = self.net3(x3)
        x4 = self.net4(x4)

        # 1 x 12 x 3
        return torch.cat([x1, x2, x3, x4], dim=2)


class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if not hasattr(self, 'noise'):
            self.noise = x.clone().detach()

        sigma = self.sigma
        noise = self.noise

        return x + noise.normal_(std=sigma)


class MinibatchStandardDeviation(nn.Module):
    def forward(self, x):
        n, c, h, w = x.size()

        y = x.std(dim=0).mean().expand(n, 1, h, w)

        x = torch.cat([x, y], dim=1)
        return x


class RaLSGANLoss(nn.Module):
    def forward(self, x1, x2):
        y1 = x1 - x2.mean()
        y2 = x2 - x1.mean()

        return torch.mean((y1 - 1) ** 2) + torch.mean((y2 + 1) ** 2)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # pretrained model architecture

        self.net = nn.Sequential(
            nn.ReflectionPad2d(40),

            nn.Conv2d(3, 32, 9, padding=4),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),

            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),

            nn.Conv2d(32, 3, 9, padding=4),
            nn.Tanh(),
        )

        mean = torch.Tensor(
            # BGR
            [[[103.939]], [[116.779]], [[123.68]]]
        )

        self.register_buffer('mu', mean)

    def forward(self, x):
        x = self.net(x - self.mu)

        # 3 x 224 x 224
        return 150 * x + self.mu


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        # convolutions
        x1 = self.net(x)

        # shaves image

        h = x.size(2)
        w = x.size(3)

        x2 = x[..., 2:h-2, 2:w-2]

        return x1 + x2


class PixelNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class TVLoss(nn.Module):
    def forward(self, x):
        diff1 = x[..., :-1, 1:] - x[..., :-1, :-1]
        diff2 = x[..., 1:, :-1] - x[..., :-1, :-1]

        # smoothness: 1e-6
        return torch.mean(torch.sqrt(diff1 ** 2 + diff2 ** 2))


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.f = nn.Conv2d(channels, channels // 8, 1)

        self.g = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.MaxPool2d(2),
        )

        self.h = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.MaxPool2d(2),
        )

        self.v = nn.Conv2d(channels // 2, channels, 1)

        tensor = torch.zeros(1)

        self.w = nn.Parameter(tensor)

    def forward(self, x):
        f = self.f(x).flatten(2)
        g = self.g(x).flatten(2)
        h = self.h(x).flatten(2)

        s = torch.bmm(f.transpose(1, 2), g)

        b = F.softmax(s, dim=-1)

        v = torch.bmm(h, b.transpose(1, 2))

        v = v.view(
            v.size(0),
            v.size(1),
            x.size(2),
            x.size(3),
        )

        o = self.v(v)

        return self.w * o + x


class DeepFaceDescriptor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # deep face recognition
        self.net = torchvision.models.vgg16()

        # removes the last classification layer
        self.net.classifier[-1] = nn.Identity()

        if pretrained:
            params = self.net.parameters()

            # 3 x 224 x 224
            state_dict = torch.hub.load_state_dict_from_url(
                'http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth')

            tensors = state_dict.values()

            for param, tensor in zip(params, tensors):
                param.data.copy_(tensor)

            mean = torch.Tensor(
                # RGB
                [[[129.186279296875]], [[104.76238250732422]], [[93.59396362304688]]]
            )

            self.register_buffer('mu', mean)

    def forward(self, x):
        # BGR -> RGB
        x = x.flip(1)

        h = self.net(x - self.mu)
        return h


def foreveriter(iterable):
    while True:
        for x in iter(iterable):
            yield x


class ReplayBuffer:
    def __init__(self, size):
        self.data = collections.deque(maxlen=size)

    def __call__(self, imgs):
        data = self.data

        if len(data) < data.maxlen:
            data.extend(imgs)
        else:
            rets = []

            for img in imgs:
                if np.random.rand() < 0.5:
                    idx = np.random.randint(data.maxlen)

                    img, data[idx] = data[idx], img

                rets.append(img)

            imgs = torch.stack(rets)

        return imgs


def cacd(span, channels_last=False):
    # eyes' center:
    #   ( 80.22166667, 106.36333333)
    #   (142.20916667, 106.28666667)
    x = []

    for path in glob.glob(f'CACD2000/{span}/*.jpg'):
        # BGR
        img = cv2.imread(path)

        if not channels_last:
            # 3 x 224 x 224
            img = np.moveaxis(img, -1, 0)

        x.append(img)

    x = np.array(x)
    return x


if __name__ == '__main__':
    gan = PAGGAN()

    x1 = cacd('21-30')
    x2 = cacd('51-60')

    gan.train(x1, x2)
