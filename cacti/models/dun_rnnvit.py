from turtle import up
from cv2 import _OutputArray_DEPTH_MASK_16F
from matplotlib.pyplot import sca
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable, grad
import functools
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import einops
from cacti.utils.utils import A, At
import math
import warnings
from torch import Tensor, einsum
from .pvt import unit_vt_b1
#from pvt import unit_vt_b1
from .builder import MODELS

def A_operator(z, Phi):
    y = torch.sum(Phi * z, 1, keepdim=True)
    return y

def At_operator(z, Phi):
    y = z * Phi
    return y


def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out

class CRNNneuro(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, res_block, forward_pro = True):
        super(CRNNneuro,self).__init__()
        
        #region left
        self.kernel_size = kernel_size
        self.forward_pro = forward_pro
        # input to fuse
        self.i2f = nn.Conv2d(input_size*3+hidden_size*2, hidden_size, kernel_size, padding=kernel_size // 2)
        #fuse to res
        basic_block = functools.partial(ResidualBlock_noBN, nf=hidden_size)
        self.recon_res = make_layer(basic_block, res_block)
        self.relu = nn.ReLU(inplace=True)
        #endregion
    
    def  forward(self, input, hidden_iteration, hidden):
        
        #region left
        _,_,T,_,_ = input.shape
        if self.forward_pro:
            f1 = input[0,:,:,:,:]
            f2 = input[1,:,:,:,:]
            f3 = input[2,:,:,:,:]
        else:
            f3 = input[0,:,:,:,:]
            f2 = input[1,:,:,:,:]
            f1 = input[2,:,:,:,:]        
        x_input = torch.cat((f1, f2, f3), dim=1)
        x_cat = torch.cat((x_input, hidden, hidden_iteration), dim=1)
        in_to_fuse = self.relu(self.i2f(x_cat))
        hidden = self.recon_res(in_to_fuse)
        #endregion
        
        return hidden

class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer

    Parameters
    --------------------
    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
               input_iteration: 5d tensor, [hidden states from previous iteration] with shape (n_seq, n_batch, hidden_size, width, height)
               test: True if in test mode, False if in train mode

    Returns
    --------------------
    output: 5d tensor, shape (n_seq, n_batch, hidden_size, width, height)
    

    """
    def __init__(self, input_size, hidden_size, kernel_size, res_block):
        super(BCRNNlayer, self).__init__()
        self.hidden_size   = hidden_size
        self.kernel_size   = kernel_size
        self.input_size    = input_size
        self.res_block     = res_block
        self.CRNN_forward  = CRNNneuro(self.input_size, self.hidden_size, self.kernel_size, self.res_block, forward_pro = True)
        self.CRNN_backward = CRNNneuro(self.input_size, self.hidden_size, self.kernel_size, self.res_block, forward_pro = False)

    def forward(self, input, input_iteration, test=False): # input: (num_seqs, batch_size, channel, width, height)
        nt, nb, nc, nx, ny = input.shape 
        size_h = [nb, self.hidden_size, nx, ny]
        if test:
            #  not tracking gradient
            with torch.no_grad():
                hid_init = Variable(torch.zeros(size_h).to(input.device))#.cuda()
        else:
            hid_init = Variable(torch.zeros(size_h).to(input.device))#.cuda()
 
        #region Bidirectional: forward and backward       
        
        output_f = []
        output_b = []
        # forward && backward
        hidden_f = hid_init
        #hidden_b = hid_init.clone()
        #expend
        input_iteration = input_iteration.view(nt, nb, self.hidden_size, nx, ny)
        input = torch.cat((input[1:2,:,:,:,:], input, input[nt-1:nt,:,:,:,:]), dim=0)
        # nt: n_seq, the number of frames(0~n_frames)
        for i in range(nt):
            hidden_f = self.CRNN_forward(input[i:i+3], input_iteration[i], hidden_f)
            output_f.append(hidden_f) #add element at the end
            

        output_f = torch.cat(output_f) 
         # backward
        hidden_b = hid_init
        # nt: n_seq, the number of frames(0~n_frames)
        for i in range(nt):
            hidden_b = self.CRNN_backward(input[nt - i - 1:nt - i + 2], input_iteration[nt - i -1], hidden_b)
            output_b.append(hidden_b)

        output_b = torch.cat(output_b[::-1])
        output = output_f.add_(output_b)
        #endregion
        if nb == 1: # nb: number of batch
            output = output.view(nt, 1, self.hidden_size, nx, ny) # view: return a reshape tensor(source tensor not be changed)

        return output

class BCRNNlayer_shared(nn.Module):
    """
    Bidirectional Convolutional RNN layer

    Parameters
    --------------------
    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
               input_iteration: 5d tensor, [hidden states from previous iteration] with shape (n_seq, n_batch, hidden_size, width, height)
               test: True if in test mode, False if in train mode

    Returns
    --------------------
    output: 5d tensor, shape (n_seq, n_batch, hidden_size, width, height)
    

    """
    def __init__(self, input_size, hidden_size, kernel_size, res_block):
        super(BCRNNlayer_shared, self).__init__()
        self.hidden_size   = hidden_size
        self.kernel_size   = kernel_size
        self.input_size    = input_size
        self.res_block     = res_block
        self.CRNN_forward  = CRNNneuro(self.input_size, self.hidden_size, self.kernel_size, self.res_block, forward_pro = True)
        #self.CRNN_backward = CRNNneuro(self.input_size, self.hidden_size, self.kernel_size, self.res_block, forward_pro = False)

    def forward(self, input, input_iteration, test=False): # input: (num_seqs, batch_size, channel, width, height)
        nt, nb, nc, nx, ny = input.shape 
        size_h = [nb, self.hidden_size, nx, ny]
        if test:
            #  not tracking gradient
            with torch.no_grad():
                hid_init = Variable(torch.zeros(size_h).to(input.device))#.cuda()
        else:
            hid_init = Variable(torch.zeros(size_h).to(input.device))#.cuda()
 
        #region Bidirectional: forward and backward       
        
        output_f = []
        output_b = []
        # forward && backward
        hidden_f = hid_init
        #hidden_b = hid_init.clone()
        #expend
        input_iteration = input_iteration.view(nt, nb, self.hidden_size, nx, ny)
        input = torch.cat((input[1:2,:,:,:,:], input, input[nt-1:nt,:,:,:,:]), dim=0)
        # nt: n_seq, the number of frames(0~n_frames)
        self.CRNN_forward.forward_pro = True
        for i in range(nt):
            hidden_f = self.CRNN_forward(input[i:i+3], input_iteration[i], hidden_f)
            output_f.append(hidden_f) #add element at the end
            

        output_f = torch.cat(output_f) 
         # backward
        hidden_b = hid_init
        # nt: n_seq, the number of frames(0~n_frames)
        self.CRNN_forward.forward_pro = False
        for i in range(nt):
            hidden_b = self.CRNN_forward(input[nt - i - 1:nt - i + 2], input_iteration[nt - i -1], hidden_b)
            output_b.append(hidden_b)

        output_b = torch.cat(output_b[::-1])
        output = output_f.add_(output_b)
        #endregion
        if nb == 1: # nb: number of batch
            output = output.view(nt, 1, self.hidden_size, nx, ny) # view: return a reshape tensor(source tensor not be changed)

        return output

class BPA_CRNN(nn.Module):
    """
    Model for Dynamic Video Super-resolution Reconstruction using Convolutional Neural Networks

    Parameters
    -----------------------
    incomings: three 5d tensors, [input_image], each of shape (batch_size, num_channels, n_seq, width, height)

    Returns
    ------------------------------
    output: 5d tensor, [output_image] with shape (batch_size, num_channels, n_seq, width, height)
    """
    def __init__(self, color_channels, num_filters, kernel_size, num_iterations,
                  num_NN_layers,res_block):
        """
        :param num_channels: number of channels
        :param num_filters: number of filters
        :param kernel_size: kernel size
        :param num_iterations: number of iterations
        :param num_NN_layers: number of CRNN/BCRNN/CNN layers in each iteration
        """
        #The super mechanism can ensure that the common parent class is executed only once. 
        # As for the execution order, it is carried out according to MRO (Method Resolution Order)
        super(BPA_CRNN, self).__init__() 
        self.num_iterations = num_iterations
        self.num_NN_layers = num_NN_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.color_channels = color_channels
        in_channels = 1
        
        self.conv0_x    = nn.ModuleList([])
        self.unit_vit_x = unit_vt_b1(out_dim=self.num_filters,pretrained=None)#nn.ModuleList([])
        self.bcrnn      = nn.ModuleList([])
        self.conv1_x    = nn.ModuleList([])
        self.conv1_h    = nn.ModuleList([])
        self.conv2_x    = nn.ModuleList([])
        self.conv2_h    = nn.ModuleList([])
        self.conv3_x    = nn.ModuleList([])
        self.conv3_h    = nn.ModuleList([])
        self.conv4_x    = nn.ModuleList([])
        # BCRNN * 5, CRNN-i * 3, # CNN * 1
        self.conv0    = nn.Conv2d(in_channels*2, 3, kernel_size, padding = kernel_size // 2)
        for i in range(num_iterations-1):
            self.conv0_x.append(
                nn.Conv2d(in_channels*2, in_channels, kernel_size, padding = kernel_size // 2),
            )
            eta_step = nn.Parameter(torch.Tensor([0.01]))
            setattr(self, f"eta_step{i}", eta_step)
        for i in range(num_iterations):
            
            self.bcrnn.append(
                BCRNNlayer(in_channels, num_filters, kernel_size, res_block),
            )
            self.conv1_x.append(
                nn.Conv2d(num_filters, num_filters, kernel_size, padding = kernel_size // 2),
            )
            self.conv1_h.append(
                nn.Conv2d(num_filters, num_filters, kernel_size, padding = kernel_size // 2),
            )
            self.conv2_x.append(
                nn.Conv2d(num_filters, num_filters, kernel_size, padding = kernel_size // 2),
            )
            self.conv2_h.append(
                nn.Conv2d(num_filters, num_filters, kernel_size, padding = kernel_size // 2),
            )
            self.conv3_x.append(
                nn.Conv2d(num_filters, num_filters, kernel_size, padding = kernel_size // 2),
            )
            self.conv3_h.append(
                nn.Conv2d(num_filters, num_filters, kernel_size, padding = kernel_size // 2),
            )
            self.conv4_x.append(
                nn.Conv2d(num_filters, in_channels, kernel_size, padding = kernel_size // 2),
            )
   
        # self.img_upsample = Upsample(input_size = num_channels, num_filters = num_filters, 
        #                               kernel_size = 3, scale_factor = 4, load_path = False)
        # self.alpha = nn.Parameter(torch.FloatTensor(1) , requires_grad = True)
        # self.beta = nn.Parameter(torch.FloatTensor(1) , requires_grad = True)
        # #init
        # self.alpha.data.fill_(0.9)
        # self.beta.data.fill_(0.1)
        #self.eta_step = nn.Parameter(torch.Tensor([0.01]))
        self.relu = nn.ReLU(inplace=True) # inplace=True: change source data, and share source address, it doesn't allocate new spance

    def forward(self, x, y, Phi, Phi_s, test=False):
        """
        x   - input in image domain, of shape (batch_size, num_channels, nx, ny, n_seq) #
        test - True: the model is in test mode, False: train mode
        """           
        if self.color_channels == 3:
            y = torch.nn.functional.pixel_shuffle(y.permute(1,0,2,3), 2).permute(1,0,2,3)
            Phi = torch.nn.functional.pixel_shuffle(Phi.permute(1,0,2,3), 2).permute(1,0,2,3)
            Phi_s = torch.nn.functional.pixel_shuffle(Phi_s.permute(1,0,2,3), 2).permute(1,0,2,3)
        y1 = torch.zeros_like(y).to(y.device)
        #Phi_s[Phi_s == 0] = 1
        y_ = y / Phi_s  # torch.Size([1, 1, 1, 128, 128])
        y_ = y_.expand_as(x.squeeze(1)).unsqueeze(1)  # torch.Size([1, 8, 1, 128, 128])
        x_batch, x_ch, x_seq, x_w, x_h = x.shape
        net = {}  
        size_h = [x_batch * x_seq, self.num_filters, x_w, x_h]
        if test:
            with torch.no_grad():
                hid_init = Variable(torch.zeros(size_h).to(y.device))#.cuda()
        else:
            hid_init = Variable(torch.zeros(size_h).to(y.device))#.cuda()
        for j in range(self.num_NN_layers-1):
            net['t0_x%d'%j]=hid_init 
        # Spatial Denoising
        z_ = x.contiguous()
        y_ = y_.contiguous().permute(0,2,1,3,4)
        z_ = z_.permute(0,2,1,3,4).view(x_batch * x_seq, x_ch, x_w, x_h)  # torch.Size([1, 1, 8, 128, 128])/torch.Size([8, 1, 8, 64, 64])
        y_ = y_.view(x_batch * x_seq, x_ch, x_w, x_h)  # torch.Size([1, 1, 8, 128, 128])

        net['t0_x0'] = self.unit_vit_x(self.relu(self.conv0(torch.cat((z_, y_), dim=1))))
        z = x
        #print("X shape:",x.shape)
        for i in range(1,self.num_iterations+1):
        
            x = x.permute(2,0,1,3,4) #x_seq, x_batch, x_ch, x_w, x_h
            x = x.contiguous()
            net['t%d_x0'%i]  = self.bcrnn[i - 1](x, net['t%d_x0'%(i-1)], test) 
            net['t%d_x0'%i]  = net['t%d_x0'%i].view(-1,self.num_filters, x_w, x_h)
            net['t%d_x1'%i]  = self.conv1_x[i - 1](net['t%d_x0'%i])
            net['t%d_h1'%i]  = self.conv1_h[i - 1](net['t%d_x1'%(i-1)])
            net['t%d_x1'%i]  = self.relu(net['t%d_h1'%i].add_(net['t%d_x1'%i]))
            net['t%d_x2'%i]  = self.conv2_x[i - 1](net['t%d_x1'%i])
            net['t%d_h2'%i]  = self.conv2_h[i - 1](net['t%d_x2'%(i-1)])
            net['t%d_x2'%i]  = self.relu(net['t%d_h2'%i].add_(net['t%d_x2'%i]))
            net['t%d_x3'%i]  = self.conv3_x[i - 1](net['t%d_x2'%i])
            net['t%d_h3'%i]  = self.conv3_h[i - 1](net['t%d_x3'%(i-1)])
            net['t%d_x3'%i]  = self.relu(net['t%d_h3'%i].add_(net['t%d_x3'%i]))

            # data consistency
            net['t%d_out'%i] = self.conv4_x[i - 1](net['t%d_x3'%i])
            net['t%d_out'%i] = net['t%d_out'%i].view(-1,x_batch, x_ch, x_w, x_h)
            net['t%d_out'%i] = net['t%d_out'%i].permute(1,2,0,3,4) # x_batch, x_ch, x_seq, x_w, x_h
            net['t%d_out'%i].contiguous()
            net['t%d_out'%i] = net['t%d_out'%i].add_(z)
            z = net['t%d_out'%i]
            if i < self.num_iterations:
                eta_step = getattr(self, f"eta_step{i-1}")               
                #B, C, D, H, W = x.shape
                if self.color_channels==3:
                    # y_bayer, Phi_bayer, Phi_s_bayer = y, Phi, Phi_s #
                    # #x = At(y_bayer,Phi_bayer)
                    # yb = A(v,Phi_bayer)
                    # bayer = [[0,0], [0,1], [1,0], [1,1]]
                    # b,f,h,w = Phi.shape
                    # y1 = y1 + (y_bayer - yb)
                    # x = v + At(torch.div(y1-yb, Phi_s_bayer + eta_step),Phi_bayer)
                    # x = rearrange(x,"(b ba) f h w->b f h w ba",b=b)
                    # x_bayer = torch.zeros(b,f,h,w).to(y.device)
                    # for ib in range(len(bayer)): 
                    #     ba = bayer[ib]
                    #     x_bayer[:,:,ba[0]::2, ba[1]::2] = x[...,ib]
                    # x = x_bayer.unsqueeze(1)
                    yb = A(z.squeeze(1),Phi)
                    y1 = y1 + (y - yb)
                    x = z.squeeze(1) + At(torch.div(y1-yb, Phi_s+ eta_step), Phi)
                    x = x.unsqueeze(1) 

                else:
                    #x = At(y,Phi)
                    yb = A(z.squeeze(1),Phi)
                    y1 = y1 + (y - yb)
                    x = z.squeeze(1) + At(torch.div(y1-yb, Phi_s+ eta_step), Phi)
                    x = x.unsqueeze(1) 
                #Phi_s[Phi_s == 0] = 1  # torch.Size([1, 1, 128, 128])
                y_ = y / Phi_s  # torch.Size([1, 1, 1, 128, 128])
                y_ = y_.expand_as(z.squeeze(1)).unsqueeze(1)  # torch.Size([1, 8, 1, 128, 128])
                z  = x #B, C, D, H, W = x.shape
                z_ = x.contiguous().view(x_batch * x_seq, x_ch, x_w, x_h)  # torch.Size([8, 1, 8, 64, 64])
                y_ = y_.contiguous().view(x_batch * x_seq, x_ch, x_w, x_h)  
                x  = self.relu(self.conv0_x[i-1](torch.cat((z_, y_), dim=1)))
                x  = x.view(x_batch, x_seq, x_ch, x_w, x_h).permute(0,2,1,3,4)
                #x_batch, x_ch, x_seq, x_w, x_h = x.shape
            # clean up i-1
            if test:
                to_delete = [ key for key in net if ('t%d'%(i-1)) in key ]

                for elt in to_delete:
                    del net[elt]

                torch.cuda.empty_cache()
                
        return [net['t%d_out'%i],net['t%d_x3'%i]]

class BPA_CRNN_shared(nn.Module):
    """
    Model for Dynamic Video Super-resolution Reconstruction using Convolutional Neural Networks

    Parameters
    -----------------------
    incomings: three 5d tensors, [input_image], each of shape (batch_size, num_channels, n_seq, width, height)

    Returns
    ------------------------------
    output: 5d tensor, [output_image] with shape (batch_size, num_channels, n_seq, width, height)
    """
    def __init__(self, color_channels, num_filters, kernel_size, num_iterations,
                  num_NN_layers,res_block):
        """
        :param num_channels: number of channels
        :param num_filters: number of filters
        :param kernel_size: kernel size
        :param num_iterations: number of iterations
        :param num_NN_layers: number of CRNN/BCRNN/CNN layers in each iteration
        """
        #The super mechanism can ensure that the common parent class is executed only once. 
        # As for the execution order, it is carried out according to MRO (Method Resolution Order)
        super(BPA_CRNN_shared, self).__init__() 
        self.num_iterations = num_iterations
        self.num_NN_layers = num_NN_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.color_channels = color_channels
        in_channels = 1
        
        self.conv0_x    = nn.Conv2d(in_channels*2, in_channels, kernel_size, padding = kernel_size // 2)
        self.unit_vit_x = unit_vt_b1(out_dim=self.num_filters,pretrained=None)
        # BCRNN * 5, CRNN-i * 3, # CNN * 1
        self.bcrnn      = BCRNNlayer(in_channels, num_filters, kernel_size, res_block)
        self.conv1_x    = nn.Conv2d(num_filters, num_filters, kernel_size, padding = kernel_size // 2)
        self.conv1_h    = nn.Conv2d(num_filters, num_filters, kernel_size, padding = kernel_size // 2)
        self.conv2_x    = nn.Conv2d(num_filters, num_filters, kernel_size, padding = kernel_size // 2)
        self.conv2_h    = nn.Conv2d(num_filters, num_filters, kernel_size, padding = kernel_size // 2)
        self.conv3_x    = nn.Conv2d(num_filters, num_filters, kernel_size, padding = kernel_size // 2)
        self.conv3_h    = nn.Conv2d(num_filters, num_filters, kernel_size, padding = kernel_size // 2)
        self.conv4_x    = nn.Conv2d(num_filters, in_channels, kernel_size, padding = kernel_size // 2)
        self.conv0    = nn.Conv2d(in_channels*2, 3, kernel_size, padding = kernel_size // 2)
        for i in range(num_iterations-1):
            eta_step = nn.Parameter(torch.Tensor([0.01]))
            setattr(self, f"eta_step{i}", eta_step)
        self.relu = nn.ReLU(inplace=True) # inplace=True: change source data, and share source address, it doesn't allocate new spance

    def forward(self, x, y, Phi, Phi_s, test=False):
        """
        x   - input in image domain, of shape (batch_size, num_channels, nx, ny, n_seq) #
        test - True: the model is in test mode, False: train mode
        """           
        if self.color_channels == 3:
            y = torch.nn.functional.pixel_shuffle(y.permute(1,0,2,3), 2).permute(1,0,2,3)
            Phi = torch.nn.functional.pixel_shuffle(Phi.permute(1,0,2,3), 2).permute(1,0,2,3)
            Phi_s = torch.nn.functional.pixel_shuffle(Phi_s.permute(1,0,2,3), 2).permute(1,0,2,3)
        y1 = torch.zeros_like(y).to(y.device)
        #Phi_s[Phi_s == 0] = 1
        y_ = y / Phi_s  # torch.Size([1, 1, 1, 128, 128])
        y_ = y_.expand_as(x.squeeze(1)).unsqueeze(1)  # torch.Size([1, 8, 1, 128, 128])
        x_batch, x_ch, x_seq, x_w, x_h = x.shape
        net = {}  
        size_h = [x_batch * x_seq, self.num_filters, x_w, x_h]
        if test:
            with torch.no_grad():
                hid_init = Variable(torch.zeros(size_h).to(y.device))#.cuda()
        else:
            hid_init = Variable(torch.zeros(size_h).to(y.device))#.cuda()
        for j in range(self.num_NN_layers-1):
            net['t0_x%d'%j]=hid_init 
        # Spatial Denoising
        z_ = x.contiguous()
        y_ = y_.contiguous().permute(0,2,1,3,4)
        z_ = z_.permute(0,2,1,3,4).view(x_batch * x_seq, x_ch, x_w, x_h)  # torch.Size([1, 1, 8, 128, 128])/torch.Size([8, 1, 8, 64, 64])
        y_ = y_.view(x_batch * x_seq, x_ch, x_w, x_h)  # torch.Size([1, 1, 8, 128, 128])

        net['t0_x0'] = self.unit_vit_x(self.relu(self.conv0(torch.cat((z_, y_), dim=1))))
        z = x
        #print("X shape:",x.shape)
        for i in range(1,self.num_iterations+1):
        
            x = x.permute(2,0,1,3,4) #x_seq, x_batch, x_ch, x_w, x_h
            x = x.contiguous()
            net['t%d_x0'%i]  = self.bcrnn(x, net['t%d_x0'%(i-1)], test) 
            net['t%d_x0'%i]  = net['t%d_x0'%i].view(-1,self.num_filters, x_w, x_h)
            net['t%d_x1'%i]  = self.conv1_x(net['t%d_x0'%i])
            net['t%d_h1'%i]  = self.conv1_h(net['t%d_x1'%(i-1)])
            net['t%d_x1'%i]  = self.relu(net['t%d_h1'%i].add_(net['t%d_x1'%i]))
            net['t%d_x2'%i]  = self.conv2_x(net['t%d_x1'%i])
            net['t%d_h2'%i]  = self.conv2_h(net['t%d_x2'%(i-1)])
            net['t%d_x2'%i]  = self.relu(net['t%d_h2'%i].add_(net['t%d_x2'%i]))
            net['t%d_x3'%i]  = self.conv3_x(net['t%d_x2'%i])
            net['t%d_h3'%i]  = self.conv3_h(net['t%d_x3'%(i-1)])
            net['t%d_x3'%i]  = self.relu(net['t%d_h3'%i].add_(net['t%d_x3'%i]))

            # data consistency
            net['t%d_out'%i] = self.conv4_x(net['t%d_x3'%i])
            net['t%d_out'%i] = net['t%d_out'%i].view(-1,x_batch, x_ch, x_w, x_h)
            net['t%d_out'%i] = net['t%d_out'%i].permute(1,2,0,3,4) # x_batch, x_ch, x_seq, x_w, x_h
            net['t%d_out'%i].contiguous()
            net['t%d_out'%i] = net['t%d_out'%i].add_(z)
            z = net['t%d_out'%i]
            if i < self.num_iterations:
                eta_step = getattr(self, f"eta_step{i-1}")               
                #B, C, D, H, W = x.shape
                if self.color_channels==3:
                    # y_bayer, Phi_bayer, Phi_s_bayer = y, Phi, Phi_s #
                    # #x = At(y_bayer,Phi_bayer)
                    # yb = A(v,Phi_bayer)
                    # bayer = [[0,0], [0,1], [1,0], [1,1]]
                    # b,f,h,w = Phi.shape
                    # y1 = y1 + (y_bayer - yb)
                    # x = v + At(torch.div(y1-yb, Phi_s_bayer + eta_step),Phi_bayer)
                    # x = rearrange(x,"(b ba) f h w->b f h w ba",b=b)
                    # x_bayer = torch.zeros(b,f,h,w).to(y.device)
                    # for ib in range(len(bayer)): 
                    #     ba = bayer[ib]
                    #     x_bayer[:,:,ba[0]::2, ba[1]::2] = x[...,ib]
                    # x = x_bayer.unsqueeze(1)
                    yb = A(z.squeeze(1),Phi)
                    y1 = y1 + (y - yb)
                    x = z.squeeze(1) + At(torch.div(y1-yb, Phi_s+ eta_step), Phi)
                    x = x.unsqueeze(1) 

                else:
                    #x = At(y,Phi)
                    yb = A(z.squeeze(1),Phi)
                    y1 = y1 + (y - yb)
                    x = z.squeeze(1) + At(torch.div(y1-yb, Phi_s+ eta_step), Phi)
                    x = x.unsqueeze(1) 
                #Phi_s[Phi_s == 0] = 1  # torch.Size([1, 1, 128, 128])
                y_ = y / Phi_s  # torch.Size([1, 1, 1, 128, 128])
                y_ = y_.expand_as(z.squeeze(1)).unsqueeze(1)  # torch.Size([1, 8, 1, 128, 128])
                z  = x #B, C, D, H, W = x.shape
                z_ = x.contiguous().view(x_batch * x_seq, x_ch, x_w, x_h)  # torch.Size([8, 1, 8, 64, 64])
                y_ = y_.contiguous().view(x_batch * x_seq, x_ch, x_w, x_h)  
                x  = self.relu(self.conv0_x(torch.cat((z_, y_), dim=1)))
                x  = x.view(x_batch, x_seq, x_ch, x_w, x_h).permute(0,2,1,3,4)
                #x_batch, x_ch, x_seq, x_w, x_h = x.shape
            # clean up i-1
            if test:
                to_delete = [ key for key in net if ('t%d'%(i-1)) in key ]

                for elt in to_delete:
                    del net[elt]

                torch.cuda.empty_cache()
                
        return [net['t%d_out'%i],net['t%d_x3'%i]]

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

@MODELS.register_module
class RNN_ViT_SCI(nn.Module):
    """
    Model for Dynamic Video Super-resolution Reconstruction using Convolutional Neural Networks

    Parameters
    -----------------------
    incomings: three 5d tensors, [input_image], each of shape (batch_size, num_channels, n_seq, width, height)

    Returns
    ------------------------------
    output: 5d tensor, [output_image] with shape (batch_size, num_channels, n_seq, width * scale_factor, height* scale_factor)
    """
    def __init__(self, color_channels=1, num_filters=64, num_iterations=8):

        super(RNN_ViT_SCI, self).__init__()
        self.num_filters=num_filters#64
        self.kernel_size=3
        self.num_iterations=num_iterations
        self.color_channels = color_channels
        in_channel = 1
        # BPA 1X
        self.bpa_crnn = BPA_CRNN(color_channels = self.color_channels, num_filters = self.num_filters,
                                    kernel_size = self.kernel_size, num_iterations = self.num_iterations, num_NN_layers = 5,
                                    res_block = 5) 
        
        #fuse cat
        self.conv1 = nn.Conv2d(in_channel  + self.num_filters, self.num_filters, self.kernel_size, padding = self.kernel_size // 2, bias = True)   
        #res
        basic_block = functools.partial(ResidualBlock_noBN, nf=self.num_filters)
        self.recon_res = make_layer(basic_block, 10)
        #self.conv2 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, padding = self.kernel_size//2)
        # output 3 channels
        self.conv3 = nn.Conv2d(self.num_filters, self.color_channels, self.kernel_size, padding = self.kernel_size//2)
        self.relu = nn.ReLU(inplace=True)

       
    def bayer_init(self,y,Phi,Phi_s):
        bayer = [[0,0], [0,1], [1,0], [1,1]]
        b,f,h,w = Phi.shape
        y_bayer = torch.zeros(b,1,h//2,w//2,4).to(y.device)
        Phi_bayer = torch.zeros(b,f,h//2,w//2,4).to(y.device)
        Phi_s_bayer = torch.zeros(b,1,h//2,w//2,4).to(y.device)
        for ib in range(len(bayer)):
            ba = bayer[ib]
            y_bayer[...,ib] = y[:,:,ba[0]::2,ba[1]::2]
            Phi_bayer[...,ib] = Phi[:,:,ba[0]::2,ba[1]::2]
            Phi_s_bayer[...,ib] = Phi_s[:,:,ba[0]::2,ba[1]::2]
        y_bayer = rearrange(y_bayer,"b f h w ba->(b ba) f h w")
        Phi_bayer = rearrange(Phi_bayer,"b f h w ba->(b ba) f h w")
        Phi_s_bayer = rearrange(Phi_s_bayer,"b f h w ba->(b ba) f h w")

        return y_bayer, Phi_bayer, Phi_s_bayer

    def forward(self, y, Phi, Phi_s, test=False):
        out_list = []
        if self.color_channels==3:
            y_bayer, Phi_bayer, Phi_s_bayer = self.bayer_init(y,Phi,Phi_s) ##B, C, D, H, W = x.shape
            x = At(y_bayer,Phi_bayer)
            yb = A(x,Phi_bayer)
            bayer = [[0,0], [0,1], [1,0], [1,1]]
            b,f,h,w = Phi.shape
            x = x + At(torch.div(y_bayer-yb,Phi_s_bayer),Phi_bayer)
            x = rearrange(x,"(b ba) f h w->b f h w ba",b=b)
            x_bayer = torch.zeros(b,f,h,w).to(y.device)
            for ib in range(len(bayer)): 
                ba = bayer[ib]
                x_bayer[:,:,ba[0]::2, ba[1]::2] = x[...,ib]
            x = x_bayer.unsqueeze(1)
            y, Phi, Phi_s = y_bayer, Phi_bayer, Phi_s_bayer
        else:
            x = At(y,Phi)
            yb = A(x,Phi)
            x = x + At(torch.div(y-yb,Phi_s),Phi)
            x = x.unsqueeze(1) ##B, C, D, H, W = x.shape

        n_batch, n_ch, n_seq, width, height = x.shape 
        x1_bpa   = self.bpa_crnn(x, y, Phi, Phi_s, test)
        img      = x1_bpa[0].contiguous()
        x1_cat   = torch.cat((img.view(-1, n_ch, width, height), x1_bpa[1].view(-1, self.num_filters, width, height)), dim=1)      
        fuse_cat = self.relu(self.conv1(x1_cat))
        res      = self.recon_res(fuse_cat)
        out_64   = res#fuse_cat + self.relu(self.conv2(res))

        out_3    = self.conv3(out_64)
        out_3    = out_3.view(n_batch, n_seq, self.color_channels, width, height).permute(0,2,1,3,4) 

        if self.color_channels!=3:
            out_3 = out_3.squeeze(1)
        out_list.append(out_3)

        return out_list

@MODELS.register_module
class RNN_ViT_SCI_shared(nn.Module):
    """
    Model for Dynamic Video Super-resolution Reconstruction using Convolutional Neural Networks

    Parameters
    -----------------------
    incomings: three 5d tensors, [input_image], each of shape (batch_size, num_channels, n_seq, width, height)

    Returns
    ------------------------------
    output: 5d tensor, [output_image] with shape (batch_size, num_channels, n_seq, width * scale_factor, height* scale_factor)
    """
    def __init__(self, color_channels=1, num_filters=64, num_iterations=8):

        super(RNN_ViT_SCI_shared, self).__init__()
        self.num_filters=num_filters#64
        self.kernel_size=3
        self.num_iterations=num_iterations
        self.color_channels = color_channels
        in_channel = 1
        # BPA 1X
        self.bpa_crnn = BPA_CRNN_shared(color_channels = self.color_channels, num_filters = self.num_filters,
                                    kernel_size = self.kernel_size, num_iterations = self.num_iterations, num_NN_layers = 5,
                                    res_block = 5) 
        
        #fuse cat
        self.conv1 = nn.Conv2d(in_channel  + self.num_filters, self.num_filters, self.kernel_size, padding = self.kernel_size // 2, bias = True)   
        #res
        basic_block = functools.partial(ResidualBlock_noBN, nf=self.num_filters)
        self.recon_res = make_layer(basic_block, 10)
        #self.conv2 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, padding = self.kernel_size//2)
        # output 3 channels
        self.conv3 = nn.Conv2d(self.num_filters, self.color_channels, self.kernel_size, padding = self.kernel_size//2)
        self.relu = nn.ReLU(inplace=True)

       
    def bayer_init(self,y,Phi,Phi_s):
        bayer = [[0,0], [0,1], [1,0], [1,1]]
        b,f,h,w = Phi.shape
        y_bayer = torch.zeros(b,1,h//2,w//2,4).to(y.device)
        Phi_bayer = torch.zeros(b,f,h//2,w//2,4).to(y.device)
        Phi_s_bayer = torch.zeros(b,1,h//2,w//2,4).to(y.device)
        for ib in range(len(bayer)):
            ba = bayer[ib]
            y_bayer[...,ib] = y[:,:,ba[0]::2,ba[1]::2]
            Phi_bayer[...,ib] = Phi[:,:,ba[0]::2,ba[1]::2]
            Phi_s_bayer[...,ib] = Phi_s[:,:,ba[0]::2,ba[1]::2]
        y_bayer = rearrange(y_bayer,"b f h w ba->(b ba) f h w")
        Phi_bayer = rearrange(Phi_bayer,"b f h w ba->(b ba) f h w")
        Phi_s_bayer = rearrange(Phi_s_bayer,"b f h w ba->(b ba) f h w")

        return y_bayer, Phi_bayer, Phi_s_bayer

    def forward(self, y, Phi, Phi_s, test=False):
        out_list = []
        if self.color_channels==3:
            y_bayer, Phi_bayer, Phi_s_bayer = self.bayer_init(y,Phi,Phi_s) ##B, C, D, H, W = x.shape
            x = At(y_bayer,Phi_bayer)
            yb = A(x,Phi_bayer)
            bayer = [[0,0], [0,1], [1,0], [1,1]]
            b,f,h,w = Phi.shape
            x = x + At(torch.div(y_bayer-yb,Phi_s_bayer),Phi_bayer)
            x = rearrange(x,"(b ba) f h w->b f h w ba",b=b)
            x_bayer = torch.zeros(b,f,h,w).to(y.device)
            for ib in range(len(bayer)): 
                ba = bayer[ib]
                x_bayer[:,:,ba[0]::2, ba[1]::2] = x[...,ib]
            x = x_bayer.unsqueeze(1)
            y, Phi, Phi_s = y_bayer, Phi_bayer, Phi_s_bayer
        else:
            x = At(y,Phi)
            yb = A(x,Phi)
            x = x + At(torch.div(y-yb,Phi_s),Phi)
            x = x.unsqueeze(1) ##B, C, D, H, W = x.shape

        n_batch, n_ch, n_seq, width, height = x.shape 
        x1_bpa   = self.bpa_crnn(x, y, Phi, Phi_s, test)
        img      = x1_bpa[0].contiguous()
        x1_cat   = torch.cat((img.view(-1, n_ch, width, height), x1_bpa[1].view(-1, self.num_filters, width, height)), dim=1)      
        fuse_cat = self.relu(self.conv1(x1_cat))
        res      = self.recon_res(fuse_cat)
        out_64   = res#fuse_cat + self.relu(self.conv2(res))

        out_3    = self.conv3(out_64)
        out_3    = out_3.view(n_batch, n_seq, self.color_channels, width, height).permute(0,2,1,3,4) 

        if self.color_channels!=3:
            out_3 = out_3.squeeze(1)
        out_list.append(out_3)

        return out_list
      
# for debug and params
if __name__ == '__main__':
    from torch.autograd import Variable, grad

    model = RNN_ViT_SCI_shared(color_channels=1, num_filters=64, num_iterations=5) 
    input,target_gnd= torch.randn(1, 1, 256, 256),torch.randn(1, 1, 1, 256, 256) # input and target are both tensor, input:[N,C,T,H,W] , target:[N,C,H,W]
    mask = torch.randn(1, 8, 256, 256)
    mask_s = torch.randn(1, 1, 256, 256)

    print(model)

    from thop import profile
    from thop import clever_format
    test=False
    macs, params = profile(model, inputs=(input, mask, mask_s, test,)) 
    
    macs, params = clever_format([macs, params], "%.5f")
    print(macs)     
    print(params)
    #RNN_ViT_SCI:5i-64ch: 7.112M, 3356G 10i-64ch: 12.671M 6266G 10i-128ch: 48.297M 24934G
    #RNN_ViT_SCI_shared:5i-64ch: 2.665M, 3356G