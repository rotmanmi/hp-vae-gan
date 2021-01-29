import argparse
import utils
import os
from glob import glob
import ast

from utils import logger, tools
import logging
import colorama

import torch
from torch.utils.data import DataLoader
import torchvision.utils
from modules import networks_new
from datasets.image import SingleImageDataset
from tkinter import *
from PIL import ImageTk, Image

root = Tk()
root.geometry("640x480")

clear = colorama.Style.RESET_ALL
blue = colorama.Fore.CYAN + colorama.Style.BRIGHT
green = colorama.Fore.GREEN + colorama.Style.BRIGHT
magenta = colorama.Fore.MAGENTA + colorama.Style.BRIGHT


def eval(opt, netG, amplitude=1):
    # Re-generate dataset frames

    if not hasattr(opt, 'Z_init_size'):
        initial_size = utils.get_scales_by_index(0, opt.scale_factor, opt.stop_scale, opt.img_size)
        initial_size = [int(initial_size * opt.ar), initial_size]
        opt.Z_init_size = [opt.batch_size, opt.latent_dim, *initial_size]

    # Parallel
    if opt.device == 'cuda':
        G_curr = torch.nn.DataParallel(netG)
    else:
        G_curr = netG

    progressbar_args = {
        "iterable": range(opt.niter),
        "desc": "Training scale [{}/{}]".format(opt.scale_idx + 1, opt.stop_scale + 1),
        "train": True,
        "offset": 0,
        "logging_on_update": False,
        "logging_on_close": True,
        "postfix": True
    }
    epoch_iterator = tools.create_progressbar(**progressbar_args)

    iterator = iter(data_loader)

    random_samples = []

    for iteration in epoch_iterator:
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(opt.data_loader)
            data = next(iterator)

        if opt.scale_idx > 0:
            real, real_zero = data
            real = real.to(opt.device)
        else:
            real = data.to(opt.device)

        noise_init = utils.generate_noise(size=opt.Z_init_size, device=opt.device)

        # Update progress bar
        epoch_iterator.set_description('Scale [{}/{}], Iteration [{}/{}]'.format(
            opt.scale_idx + 1, opt.stop_scale + 1,
            iteration + 1, opt.niter,
        ))
        G_curr.eval()
        import numpy as np
        import sys
        with torch.no_grad():
            fake_var = []
            fake_vae_var = []
            for _ in range(opt.num_samples):
                noise_init = utils.generate_noise(ref=noise_init)
                channel_idxs = np.random.choice(np.arange(0, 128), 127, replace=False)
                # U = torch.zeros(1, 128, 5).normal_(0, 1).to(noise_init.device)
                U = torch.zeros(1, 128, 1).to(noise_init.device)
                U[:, _] = 4
                # U[:, :120] =
                V = torch.zeros(1, 1, 22, 33).to(noise_init.device)
                # V.bernoulli_(p=0.01)
                V[:, :, 1:4, 20:32] = amplitude
                # V[:, :, 4:10, 8:10] = 1
                V = V.flatten(2)
                UV = torch.bmm(U, V).view(1, 128, 22, 33)
                UV = (UV - UV.mean()) / UV.std()
                # noise_init[:] = 0
                # noise_init[:, :, 5:11, 16:18] = _
                # noise_init[:, 108, 0:4, 0:4] = 100
                # noise_init[:, 21, _:_ + 1, 16:19] = 0.01
                # noise_init[:, :, 3:11, 16:18] = -10 / opt.num_samples

                # normed_z_vae = z_vae / ((z_vae ** 2).sum() + sys.float_info.epsilon)
                # noise_init = noise_init / ((noise_init ** 2).sum() + sys.float_info.epsilon)
                noise_init = UV
                fake, fake_vae = G_curr(noise_init, opt.Noise_Amps, noise_init=noise_init, mode="none")
                fake_var.append(fake)
                fake_vae_var.append(fake_vae)
            fake_var = torch.cat(fake_var, dim=0)
            fake_vae_var = torch.cat(fake_vae_var, dim=0)

        opt.summary.visualize_image(opt, iteration, real, 'Real')
        opt.summary.visualize_image(opt, iteration, fake_var, 'Fake var')
        opt.summary.visualize_image(opt, iteration, fake_vae_var, 'Fake VAE var')

        random_samples.append(fake_var)

    random_samples = torch.cat(random_samples, dim=0)
    from torchvision.utils import save_image
    save_image(random_samples, 'test.png', normalize=True)
    torch.save(random_samples, os.path.join(opt.saver.eval_dir, "random_samples.pth"))
    epoch_iterator.close()

    return torchvision.transforms.ToPILImage()(
        torchvision.utils.make_grid(fake_var[:3, :, :, :].clone().cpu().data, 3, normalize=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-dir', required=True, help="Experiment directory")
    parser.add_argument('--num-samples', type=int, default=1, help='number of samples to generate')
    parser.add_argument('--netG', default='netG.pth', help="path to netG (to continue training)")
    parser.add_argument('--niter', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--data-rep', type=int, default=1, help='data repetition')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables cuda')

    parser.set_defaults(hflip=False)
    opt = parser.parse_args()

    exceptions = ['no-cuda', 'niter', 'data_rep', 'batch_size', 'netG']
    all_dirs = glob(opt.exp_dir)

    progressbar_args = {
        "iterable": all_dirs,
        "desc": "Experiments",
        "train": True,
        "offset": 0,
        "logging_on_update": False,
        "logging_on_close": True,
        "postfix": True
    }
    exp_iterator = tools.create_progressbar(**progressbar_args)

    for idx, exp_dir in enumerate(exp_iterator):
        opt.experiment_dir = exp_dir
        keys = vars(opt).keys()
        with open(os.path.join(exp_dir, 'args.txt'), 'r') as f:
            for line in f.readlines():
                log_arg = line.replace(' ', '').replace('\n', '').split(':')
                assert len(log_arg) == 2
                if log_arg[0] in exceptions:
                    continue
                try:
                    setattr(opt, log_arg[0], ast.literal_eval(log_arg[1]))
                except Exception:
                    setattr(opt, log_arg[0], log_arg[1])

        opt.netG = os.path.join(exp_dir, opt.netG)
        if not os.path.exists(opt.netG):
            logging.info('Skipping {}, file not exists!'.format(opt.netG))
            continue

        # Define Saver
        opt.saver = utils.ImageSaver(opt)

        # Define Tensorboard Summary
        opt.summary = utils.TensorboardSummary(opt.saver.eval_dir)

        # Logger
        logger.configure_logging(os.path.abspath(os.path.join(opt.experiment_dir, 'logbook.txt')))

        # CUDA
        device = 'cuda' if torch.cuda.is_available() and not opt.no_cuda else 'cpu'
        opt.device = device
        if torch.cuda.is_available() and device == 'cpu':
            logging.info("WARNING: You have a CUDA device, so you should probably run with --cuda")

        # Adjust scales
        utils.adjust_scales2image(opt.img_size, opt)

        # Initial parameters
        opt.scale_idx = 0
        opt.nfc_prev = 0
        opt.Noise_Amps = []

        # Date
        dataset = SingleImageDataset(opt)
        data_loader = DataLoader(dataset,
                                 shuffle=True,
                                 drop_last=True,
                                 batch_size=opt.batch_size,
                                 num_workers=2)

        opt.dataset = dataset
        opt.data_loader = data_loader

        # Current networks
        assert hasattr(networks_new, opt.generator)
        netG = getattr(networks_new, opt.generator)(opt).to(opt.device)

        if not os.path.isfile(opt.netG):
            raise RuntimeError("=> no <G> checkpoint found at '{}'".format(opt.netG))
        checkpoint = torch.load(opt.netG, map_location='cpu')
        opt.scale_idx = checkpoint['scale']
        opt.resumed_idx = checkpoint['scale']
        opt.resume_dir = '/'.join(opt.netG.split('/')[:-1])
        for _ in range(opt.scale_idx):
            netG.init_next_stage()
        netG.load_state_dict(checkpoint['state_dict'])
        netG = netG.cuda()
        # NoiseAmp
        opt.Noise_Amps = torch.load(os.path.join(opt.resume_dir, 'Noise_Amps.pth'))['data']
        print(type(eval(opt, netG)))

        f = Frame(root)
        f.grid(row=1, column=4)
        image_no = ImageTk.PhotoImage(eval(opt, netG))
        label_img = Label(image=image_no)
        label_img.grid(row=3, column=0, columnspan=1)


        def callback_scale(e):
            new_image = ImageTk.PhotoImage(eval(opt, netG, amplitude_var.get()))
            label_img.configure(image=new_image)
            label_img.image = new_image


        amplitude_var = IntVar()
        amplitude_var.set(1)
        amplitude = Scale(root, from_=0, to=50, tickinterval=10, orient=HORIZONTAL, length=300,
                          variable=amplitude_var,
                          command=callback_scale, label='Channel Amplitude')

        x_var = IntVar()
        x_var.set(1)
        x = Scale(root, from_=0, to=50, tickinterval=10, orient=HORIZONTAL, length=300,
                  variable=x_var,
                  command=callback_scale, label='X')

        y_var = IntVar()
        y_var.set(1)
        y = Scale(root, from_=0, to=50, tickinterval=10, orient=HORIZONTAL, length=300,
                  variable=y_var,
                  command=callback_scale, label='Y')

        amplitude.grid(row=0, column=0)
        x.grid(row=1, column=0)
        y.grid(row=2, column=0)

        root.focus_set()
        root.mainloop()
        # eval(opt, netG)
