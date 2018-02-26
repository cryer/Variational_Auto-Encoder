import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
from config import Config
import model


def train(**kwargs):
    cfg = Config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)

    dataset = datasets.MNIST(root=cfg.train_path,
                             train=True,
                             transform=transforms.ToTensor(),
                             download=cfg.download)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=cfg.batch_size,
                                              shuffle=cfg.shuffle,
                                              num_workers=cfg.num_workers)
    vae = model.VAE(cfg)
    if cfg.use_gpu:
        vae.cuda()

    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.lr)
    data_iter = iter(data_loader)
    fixed_x, _ = next(data_iter)
    torchvision.utils.save_image(fixed_x.cpu(), './data/real_images.png')
    fixed_x = Variable(fixed_x.view(fixed_x.size(0), -1))
    if cfg.use_gpu:
        fixed_x = fixed_x.cuda()

    plt.ion()
    for epoch in range(cfg.epoch):
        for i, (images, _) in enumerate(data_loader):

            images = Variable(images.view(images.size(0), -1))
            if cfg.use_gpu:
                images = images.cuda()
            out, mean, log_var = vae(images)
            reconst_loss = F.binary_cross_entropy(out, images, size_average=False)
            kl_divergence = torch.sum(0.5 * (mean ** 2 + torch.exp(log_var) - log_var - 1))

            total_loss = reconst_loss + kl_divergence
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if i % 100 == 0:
                plt.cla()
                plt.subplot(1, 2, 1)
                plt.imshow(images.data[0].view(28, 28).cpu().numpy(), cmap="gray")

                plt.subplot(1, 2, 2)
                plt.imshow(out.data[0].view(28, 28).cpu().numpy(), cmap="gray")
                plt.draw()
                plt.pause(0.01)
        reconst_images, _, _ = vae(fixed_x)
        reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
        torchvision.utils.save_image(reconst_images.data.cpu(),
                                     './data/reconst_images_%d.png' % (epoch + 1))
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    import fire
    fire.Fire()
