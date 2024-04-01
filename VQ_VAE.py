import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.data import DataLoader

import torch.optim as optim

from ST_Transformer_new import STTransformer
from lib.utils import image_to_patches



class VectorQuantizer(nn.Module):
    # def __init__(self, num_embeddings=8, embedding_dim=32, commitment_cost):
    def __init__(self, num_embeddings=8, embedding_dim=16, commitment_scalar=1.0):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_scalar = commitment_scalar
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        # self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # inputs shape should be [B,1, 32, T-1]
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, self._embedding_dim)

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_scalar * e_latent_loss
        if self.training:
            quantized = inputs + (quantized - inputs).detach()
    
        return encoding_indices, quantized.contiguous(), loss


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, embed_size, time_num, num_blocks, T_dim, output_T_dim, heads, cheb_K, forward_expansion,
                 dropout, N_in, N_out):
        super(Encoder, self).__init__()

        A = torch.randn(N_in, N_in)

        self._st_transformer = STTransformer(
            A,
            in_channels, 
            embed_size, 
            time_num, 
            num_blocks, 
            T_dim, 
            output_T_dim, 
            heads,
            cheb_K,
            forward_expansion,
            dropout, 
            out_channels,
            N_in,
            N_out
        )

    def forward(self, inputs):
        # image_patches = image_to_patches(inputs)    # [B, N, C, T]

        a_p = self._st_transformer(inputs)   # predict latent actions

        return a_p
    


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, embed_size, time_num, num_blocks, T_dim, output_T_dim, heads, cheb_K, forward_expansion,
                 dropout, N_in, N_out):
        super(Decoder, self).__init__()
        A = torch.randn(N_in, N_in)

        self._st_transformer = STTransformer(
            A,
            in_channels, 
            embed_size, 
            time_num, 
            num_blocks, 
            T_dim, 
            output_T_dim, 
            heads,
            cheb_K,
            forward_expansion,
            dropout, 
            out_channels,
            N_in,
            N_out
        )

    def forward(self, aq):
        
        return self._st_transformer(aq)
    



class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()
        self._config = config
        # import ipdb; ipdb.set_trace()

        self._encoder = Encoder(config.Encoder['in_channels'], 
                                config.Encoder['out_channels'], 
                                config.Encoder['embed_size'], 
                                config.Encoder['time_num'], 
                                config.Encoder['num_blocks'],
                                config.Encoder['T_dim'], 
                                config.Encoder['output_T_dim'], 
                                config.Encoder['heads'], 
                                config.Encoder['cheb_K'], 
                                config.Encoder['forward_expansion'], 
                                config.Encoder['dropout'], 
                                config.Encoder['N_in'], 
                                config.Encoder['N_out'])
        
        self._vq_vae = VectorQuantizer(config.vq['num_embeddings'], config.vq['embedding_dim'], config.vq['commitment_scalar'])

        self._decoder = Decoder(config.Decoder['in_channels'], 
                                config.Decoder['out_channels'], 
                                config.Decoder['embed_size'], 
                                config.Decoder['time_num'], 
                                config.Decoder['num_blocks'],
                                config.Decoder['T_dim'], 
                                config.Decoder['output_T_dim'], 
                                config.Decoder['heads'], 
                                config.Decoder['cheb_K'], 
                                config.Decoder['forward_expansion'], 
                                config.Decoder['dropout'], 
                                config.Decoder['N_in'], 
                                config.Decoder['N_out'])
        
        self._sigmoid = torch.sigmoid

    def forward(self, x):
        a_p = self._encoder(x)  # action encoder
        index, a_q, vq_loss = self._vq_vae(a_p)   # quantilized latent actions
        x_his = x[:, :, :, :-1] # previous frames, shape as [B, C, N, T-1]
        ax = torch.cat((x_his, a_q.permute(0, 2, 1, 3)), dim=1)
        x_recon = self._decoder(ax) # predict last frame, input is [B, C, N, T]
        x_recon_sig = self._sigmoid(x_recon)
        return a_q, x_recon_sig, index, vq_loss

