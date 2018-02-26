
class Config(object):
    train_path='data/'
    img_size = 784
    h_dim = 400
    z_dim=20
    download = True
    batch_size=100
    shuffle = True
    num_workers = 2
    lr=1e-3
    use_gpu=True
    epoch = 20