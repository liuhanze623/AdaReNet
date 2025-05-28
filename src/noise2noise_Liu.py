import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

from unet_Liu import UNet
from utils import *
import os
import json
from torchvision.utils import save_image

class Noise2Noise(object):
    """Implementation of Noise2Noise from Lehtinen et al. (2018)."""

    def __init__(self, params, trainable):
        """Initializes model."""

        self.p = params
        self.trainable = trainable
        self._compile()

    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

        print('Noise2Noise: Learning Image Restoration without Clean Data (Lethinen et al., 2018)')

        # Model (3x3=9 channels for Monte Carlo since it uses 3 HDR buffers)
        if self.p.noise_type == 'mc':
            self.is_mc = True
            self.model = UNet(in_channels=9)
        else:
            self.is_mc = False
            self.model = UNet(in_channels=3)

        # Set optimizer and loss, if in training mode
        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=self.p.learning_rate,
                              betas=self.p.adam[:2],
                              eps=self.p.adam[2])

            
            if self.p.loss == 'hdr':
                assert self.is_mc, 'Using HDR loss on non Monte Carlo images'
                self.loss = HDRLoss()
            elif self.p.loss == 'l2':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()

        # CUDA support
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.trainable:
                self.loss = self.loss.cuda()
    
            
    def adjust_learning_rate(self, iteration, total_iterations):
        ramp_down_perc = 0.3
        initial_lr = 0.001
        ramp_down_start_iter = total_iterations * (1 - ramp_down_perc)
        
        if iteration >= ramp_down_start_iter:
            t = (iteration - ramp_down_start_iter) / (ramp_down_perc * total_iterations)
            lr = initial_lr * (0.5 + np.cos(t * np.pi) / 2) ** 2
        else:
            lr = initial_lr

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr


    def _print_params(self):
        """Formats parameters to print when training."""

        print('Training parameters: ')
        self.p.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()


    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            if self.p.clean_targets:
                ckpt_dir_name = f'{datetime.now():{self.p.noise_type}-clean-%H%M}'
            else:
                ckpt_dir_name = f'{datetime.now():{self.p.noise_type}-%H%M}'
            if self.p.ckpt_overwrite:
                if self.p.clean_targets:
                    ckpt_dir_name = f'{self.p.noise_type}-clean'
                else:
                    ckpt_dir_name = self.p.noise_type

            self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)
            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            fname_unet = '{}/n2n-{}.pt'.format(self.ckpt_dir, self.p.noise_type)
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/n2n-epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss)
        print('Saving checkpoint to: {}\n'.format(fname_unet))
        #self.model.train()
        torch.save(self.model.state_dict(), fname_unet)

        # Save stats to JSON
        fname_dict = '{}/n2n-stats.json'.format(self.ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)


    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""

        print('Loading checkpoint from: {}'.format(ckpt_fname))
        if self.use_cuda:
            self.model.load_state_dict(torch.load(ckpt_fname))
        else:
            self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))


    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        """Tracks and saves starts after each epoch."""

        # Evaluate model on validation set
        print('\rTesting model on validation set... ', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr = self.eval(valid_loader)
        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)


        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)
        self.save_model(epoch, stats, epoch == 0)

        # Plot stats
        if self.p.plot_stats:
            loss_str = f'{self.p.loss.upper()} loss'
            plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], loss_str)
            plot_per_epoch(self.ckpt_dir, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')


    def test(self, test_loader, show):
        """Evaluates denoiser on test set."""

        self.model.train(False)

        source_imgs = []
        denoised_imgs = []
        clean_imgs = []

        # Create directory for denoised images
        denoised_dir = os.path.dirname(self.p.data)
        save_path = os.path.join(denoised_dir, 'denoised')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        for batch_idx, (source, target) in enumerate(test_loader):
            # Only do first <show> images
            if show == 0 or batch_idx >= show:
                break

            source_imgs.append(source)
            clean_imgs.append(target)

            for i in range(source.size(0)):
                
                single_source = source[i, :, :, :]
                single_source_np = single_source.numpy()                
                H, W = single_source_np.shape[1], single_source_np.shape[2]
                val_size = (max(H, W) + 31) // 32 * 32

                single_source_padded = np.pad(
                    single_source_np,
                    [[0, 0], [0, val_size - H], [0, val_size - W]],
                    'reflect')

                single_source_tensor = torch.from_numpy(single_source_padded)

                single_source_tensor = single_source_tensor.unsqueeze(0)

                if self.use_cuda:
                    single_source_tensor = single_source_tensor.cuda()

                denoised_img = self.model(single_source_tensor).detach()

                denoised_img = denoised_img[:, :, :H, :W]

                denoised_imgs.append(denoised_img)

        # Squeeze tensors
        source_imgs = [t.squeeze(0) for t in source_imgs]
        denoised_imgs = [t.squeeze(0) for t in denoised_imgs]
        clean_imgs = [t.squeeze(0) for t in clean_imgs]

        # Create montage and save images
        print('Saving images and montages to: {}'.format(save_path))
        add_psnr = 0
        avg_psnr = 0
        add_ssim = 0
        avg_ssim = 0
        for i in range(len(source_imgs)):
            img_name = test_loader.dataset.imgs[i]
            a , b= create_montage(img_name, self.p.noise_type, save_path, source_imgs[i], denoised_imgs[i], clean_imgs[i], show)
            add_psnr = add_psnr + a
            add_ssim = add_ssim + b
        avg_psnr = add_psnr/len(source_imgs)
        avg_ssim = add_ssim/len(source_imgs)
        print("psnr:",add_psnr,"average psnr:",avg_psnr)
        print("ssim:",add_ssim,"average ssim:",avg_ssim)
        return avg_psnr, avg_ssim



        
    def eval(self, valid_loader):
        """Evaluates denoiser on validation set."""

        self.model.train(False)

        valid_start = datetime.now()
        loss_meter = AvgMeter()
        psnr_meter = AvgMeter()

        for batch_idx, (source, target) in enumerate(valid_loader):
            if self.use_cuda:
                source = source.cuda()
                target = target.cuda()

            # Denoise
            source_denoised = self.model(source)

            loss = self.loss(source_denoised, target)
            loss_meter.update(loss.item())

            if self.is_mc:
                source_denoised = reinhard_tonemap(source_denoised)
            # TODO: Find a way to offload to GPU, and deal with uneven batch sizes
            for i in range(self.p.batch_size):
                source_denoised = source_denoised.cpu()
                target = target.cpu()
                psnr_meter.update(psnr(source_denoised[i], target[i]).item())

        valid_loss = loss_meter.avg
        valid_time = time_elapsed_since(valid_start)[0]
        psnr_avg = psnr_meter.avg

        return valid_loss, valid_time, psnr_avg
        
        
    



    def train(self, train_loader, valid_loader):
        """Trains denoiser on training set."""

        self.model.train(True)

        self._print_params()
        num_batches = len(train_loader)
        total_iterations = self.p.nb_epochs * num_batches
        assert num_batches % self.p.report_interval == 0, 
        # Dictionaries of tracked stats
        stats = {'noise_type': self.p.noise_type,
                 'noise_param': self.p.noise_param,
                 'train_loss': [],
                 'valid_loss': [],
                 'valid_psnr': []}

        # Main training loop
        train_start = datetime.now()

        for epoch in range(self.p.nb_epochs):
                                   
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))

            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()
            

            for batch_idx, (source, target) in enumerate(train_loader):
                iteration = epoch * num_batches + batch_idx
                self.adjust_learning_rate(iteration, total_iterations) 
                batch_start = datetime.now()
                progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)
                
                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()

                # Denoise image
                source_denoised = self.model(source)

                loss = self.loss(source_denoised, target)
                loss_meter.update(loss.item())

                self.optim.zero_grad()             
                                      
                        
                loss.backward()
                self.optim.step()

                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                    show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()

            self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
            train_loss_meter.reset()

        train_elapsed = time_elapsed_since(train_start)[0]
        print('Training done! Total elapsed time: {}\n'.format(train_elapsed))


class HDRLoss(nn.Module):
    """High dynamic range loss."""

    def __init__(self, eps=0.01):
        """Initializes loss with numerical stability epsilon."""

        super(HDRLoss, self).__init__()
        self._eps = eps


    def forward(self, denoised, target):
        """Computes loss by unpacking render buffer."""

        loss = ((denoised - target) ** 2) / (denoised + self._eps) ** 2
        return torch.mean(loss.view(-1))
