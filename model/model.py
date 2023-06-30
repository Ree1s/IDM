from enum import auto
import logging
from collections import OrderedDict
from data.util import *
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
import random
import data.util as Util
from .crop_validation import forward_crop
logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None
        self.device = torch.device(
            'cuda')
        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.sub, self.div = torch.FloatTensor([0.5]).view(1, -1, 1, 1), torch.FloatTensor([0.5]).view(1, -1, 1, 1)
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        

        data['inp'] = (data['inp'] -self.sub) / self.div
        data['gt'] = (data['gt'] -self.sub) / self.div
        p = random.random()

        img_lr, img_hr = data['inp'], data['gt']
        w_hr = round(img_lr.shape[-1] + (img_hr.shape[-1] - img_lr.shape[-1]) * p)
        img_hr = resize_fn(img_hr, w_hr)
        hr_coord, _ = Util.to_pixel_samples(img_hr)
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]
        hr_coord = hr_coord.repeat(img_hr.shape[0], 1, 1)
        cell = cell.repeat(img_hr.shape[0], 1, 1)
        
        data = {
        'inp': img_lr,
        'coord': hr_coord,
        'cell': cell,
        'gt': img_hr,
        'scaler': torch.from_numpy(np.array([p], dtype=np.float32)) } 

        self.data = self.set_device(data)

    def optimize_parameters(self, scaler):
        self.optG.zero_grad()


        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['gt'].shape
        l_pix = l_pix.sum()/int(b*c*h*w)

        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, crop=False, continous=False, use_ddim=False):
        self.netG.eval()
        if crop == False:
            with torch.no_grad():
                if isinstance(self.netG, nn.parallel.DistributedDataParallel):
                    self.SR = self.netG.module.super_resolution(
                        self.data, continous, use_ddim)
                else:
                    self.SR = self.netG.super_resolution(
                        self.data, continous, use_ddim)
        else:
            with torch.no_grad():
                if isinstance(self.netG, nn.parallel.DistributedDataParallel):
                    self.SR = forward_crop(
                        self.data, self.netG.module, continous, use_ddim
                    )
                else:
                    self.SR = forward_crop(
                        self.data, self.netG, continous, use_ddim
                    )
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.parallel.DistributedDataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.parallel.DistributedDataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.parallel.DistributedDataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['inp'].detach().float().cpu()
            out_dict['HR'] = self.data['gt'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['inp'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.parallel.DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step, best=None):
        if best is not None:
            gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'best_{}_gen.pth'.format(best))
            opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'best_{}_opt.pth'.format(best))
        else:
            gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'latest_gen.pth'.format(iter_step, epoch))
            opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'latest_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.parallel.DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            if not os.path.isfile(gen_path):
                return
            # gen
            network = self.netG
            if isinstance(self.netG, nn.parallel.DistributedDataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path, map_location=torch.device('cpu')), strict=True)

            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path, map_location=torch.device('cpu'))

                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
