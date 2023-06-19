#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import torch
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


import numpy as np

def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)



# class UnrollMax(nn.Module):
#     def __init__(self,):
#         super(UnrollMax, self).__init__()

#     def forward(self, centre, neighbour, inrel):
#         concat = torch.zeros_like(centre)
#         #print('new')
#         for i in range(6):#
#             #concat += checkpoint(self.inrel,centre+torch.roll(neighbour,int((i%2-.5)*2),dims=int(i//2)+2))/6
#             concat = torch.maximum(concat,checkpoint(inrel,centre+torch.roll(neighbour,int((i%2-.5)*2),dims=int(i//2)+2)))
#         return concat



class XEdgeConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super(XEdgeConv3d, self).__init__(in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.mid_channels = max(8,(in_channels+out_channels)//2)
        self.conv1 = nn.Conv3d(in_channels, self.mid_channels, 1, padding=(0,0,0)) # fully connected 64 to 96 channels
        self.conv2 = nn.Conv3d(in_channels, self.mid_channels, 1, padding=(0,0,0)) # fully connected 64 to 96 channels
        self.conv3 = nn.Conv3d(self.mid_channels, out_channels, 1, padding=(0,0,0)) # fully connected 96 to 128 channels
        self.inrel = nn.Sequential(nn.InstanceNorm3d(self.mid_channels), nn.ReLU())

        self.t6 = torch.zeros(6,1,3,3,3).cuda()
        six = [[2,1,1],[0,1,1],[1,2,1],[1,0,1],[1,1,2],[1,1,0]]

        for i,s in enumerate(six):
            self.t6[i,0,s[0],s[1],s[2]] = 1

    def forward(self, x):
        mid = self.mid_channels
        B,C_in,H,W,D = x.shape

        centre = self.conv1(x) # Fully connected output increased to 96 channels
        neighbour = self.conv2(x) # Fully connected output increased to 96 channels

        repeated_nb_selector_kernel = self.t6.repeat(mid,1,1,1,1) # Shape: 6*96,1,3,3,3

        # Convolve with the six-neighbourhood kernel (gather) for every mid channel
        # Grouped convolutions: The more groups, the "smaller" the needed kernel channel size C/n_groups
        selected_neighbours = F.conv3d(neighbour, repeated_nb_selector_kernel, padding=1, groups=mid) # Shape: B,6*96,H,W,D
        repeated_centers = centre.unsqueeze(2).repeat(1,1,6,1,1,1) # Shape: B,96,6,H,W,D
        flattened_centers = nn.Flatten(1,2)(repeated_centers) # Shape: B,6*96,H,W,D -> Make centers ready sum with selected neighbours

        added = selected_neighbours + flattened_centers # Message passing
        added = added.view(B,mid,6*H,W,D) # Put neighbour channels into spatial dimension for instance normalisation
        nonlinear = self.inrel(added) #Shape: B,96,6*H,W,D
        aggregated = nonlinear.view(B,mid,6,H,W,D).max(2)[0] # Shape: 2,96,H,W,D

        output = self.conv3(aggregated) # Fully connected layer increased to 128 channels
        if(self.stride[0]==2):
            output = F.avg_pool3d(output,kernel_size=(2,2,2),stride=(2,2,2))
        return output

class nnUNetTrainerV2_XEdgeConvMax(nnUNetTrainerV2):
    def initialize_network(self):
        self.base_num_features = 16  # otherwise we run out of VRAM
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU

        # Configure symmetric kernel sizes
        self.net_num_pool_op_kernel_sizes = np.array(self.net_num_pool_op_kernel_sizes)
        self.net_pool_per_axis = np.array(self.net_pool_per_axis)

        for stage_idx, kernel_sizes_per_stage in enumerate(np.array(self.net_num_pool_op_kernel_sizes)):
            self.net_num_pool_op_kernel_sizes[stage_idx] = np.min(kernel_sizes_per_stage)

        self.net_pool_per_axis[:] = self.net_pool_per_axis.min()

        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    2, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)

        count = 0; count2 = 0
        for name, module in self.network.named_modules():
            if isinstance(module, nn.Conv3d):
                before = get_layer(self.network, name)
                if(before.kernel_size[0] == 3):
                    after = XEdgeConv3d(before.in_channels,before.out_channels,before.kernel_size,stride=before.stride)
                    set_layer(self.network, name, after); count += 1

            if isinstance(module, nn.ConvTranspose3d):
                before = get_layer(self.network, name)
                after = nn.Sequential(nn.Conv3d(before.in_channels,before.out_channels,1,bias=False),nn.Upsample(scale_factor=before.kernel_size,mode='trilinear',align_corners=False))
                set_layer(self.network, name, after); count2 += 1
        print(count,'# Conv3d > XEdgeConv','and',count2,'#ConvTransp')

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
