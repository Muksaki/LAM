import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import math
import numpy as np
import pandas as pd

import sys
sys.path.append('./lib/')
from pkl_process import *
from utils import load_graphdata_channel_my, compute_val_loss_sttn, image_to_patches
import tools

from time import time
import shutil
import argparse
import configparser
from tensorboardX import SummaryWriter
import os
import pathlib
import ruamel.yaml as yaml

from ST_Transformer_new import STTransformer # STTN model with linear layer to get positional embedding
from ST_Transformer_new_sinembedding import STTransformer_sinembedding #STTN model with sin()/cos() to get positional embedding, the same as "Attention is all your need"
from VQ_VAE import VQVAE
#%%


def main(config):
    params_path = config.logdir ## Path for saving network parameters
    print('params_path:', params_path)

    dataset_path = config.dataset_path
    train_episodes = tools.load_episodes(dataset_path)
    generator = tools.sample_episodes(
        train_episodes, config.batch_length
    )
    train_dataset = tools.from_generator(generator, config.batch_size)
    
    ### Construct Network
    net =  VQVAE(config)
    
    net.to(config.device)
    
    ### Training Process
    #### Load the parameter we have already learnt if start_epoch does not equal to 0
    start_epoch = 0
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    #### Loss Function Setting
    criterion = nn.MSELoss().to(config.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)    
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01)
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    losses = {}
    print(net) 
    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    start_time = time()
    if start_epoch > 0:
        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)
        net.load_state_dict(torch.load(params_filename))
        print('start epoch:', start_epoch)
        print('load weight from: ', params_filename)
    #### train model
    for epoch in range(start_epoch, config.epochs):
        
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
        net.train() 
        patch_images = image_to_patches(torch.Tensor(next(train_dataset)['image']).to(config.device))
        optimizer.zero_grad()  
        _, output, _, vq_loss = net(patch_images.permute(0, 2, 1, 3))
        losses['vq_loss'] = vq_loss
        # import ipdb; ipdb.set_trace()
        recon_loss = criterion(output.squeeze(-1), patch_images[:, :, :, -1])
        losses['recon_loss'] = recon_loss

        model_loss = sum(losses.values())
        model_loss.backward()
        optimizer.step()

        # training_loss = model_loss.item()
        for key, value in losses.items():
            sw.add_scalar(key, value, epoch)
        if epoch % config.log_every == 0:
            print('Epoch: %s, training loss: %.2f, time: %.2fs' % (epoch, model_loss, time() - start_time))
            torch.save(net.state_dict(), params_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
    defaults = {}
    for name in args.configs:
        defaults.update(configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))