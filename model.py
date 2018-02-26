import torch
import torch.nn as nn
from torch.autograd import Variable

class VAE(nn.Module):

    def __init__(self, cfg):
        super(VAE, self).__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(cfg.img_size, cfg.h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(cfg.h_dim, cfg.z_dim * 2))

        self.decoder = nn.Sequential(
            nn.Linear(cfg.z_dim, cfg.h_dim),
            nn.ReLU(),
            nn.Linear(cfg.h_dim, cfg.img_size),
            nn.Sigmoid())

    def reparameterize(self, mean, log_var):
        samples = Variable(torch.randn(mean.size(0), mean.size(1)))
        if self.cfg.use_gpu:
            samples = samples.cuda()
        z = mean + samples * torch.exp(log_var / 2)
        return z

    def forward(self, x):
        h = self.encoder(x)
        mean, log_var = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mean, log_var)
        out = self.decoder(z)
        return out, mean, log_var

    def sample(self, z):
        return self.decoder(z)
