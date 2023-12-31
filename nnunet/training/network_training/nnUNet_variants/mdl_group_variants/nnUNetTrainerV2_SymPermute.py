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
import numpy as np
from pathlib import Path
import os

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

class PermuteConv3D(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None, sym_range='full'):
        super(PermuteConv3D, self).__init__(in_channels, out_channels, kernel_size, stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Conv3d(in_channels, out_channels, (4,1,1), groups=groups).weight
        self.kernel_idxs = torch.tensor(
             [[[3,0,3],[0,1,0],[3,0,3]],\
              [[0,1,0],[1,2,1],[0,1,0]],\
              [[3,0,3],[0,1,0],[3,0,3]]])
        self.kernel_six_nb_mask = torch.tensor(
             [[[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]],\
              [[0.,1.,0.],[1.,1.,1.],[0.,1.,0.]],\
              [[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]]])

        self.sym_range = sym_range

    def forward(self, x):
        reflect_weight = self.weight[:,:,self.kernel_idxs].squeeze(6).squeeze(5)
        if self.sym_range == 'full':
            pass
        elif self.sym_range == 'six-neighbourhood-only':
            reflect_weight = reflect_weight * self.kernel_six_nb_mask.view(1,1,3,3,3).to(device=x.device)
        else:
            raise ValueError()

        x = torch.nn.functional.conv3d(x, reflect_weight,
            bias=self.bias, padding=self.padding, dilation=self.dilation,
            groups=self.groups, stride=self.stride)

        return x



class nnUNetTrainerV2_SymPermute(nnUNetTrainerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sym_range = os.environ.get('SYM_PERMUTE_RANGE', 'full')
        assert self.sym_range in ['six-neighbourhood-only', 'full']

        self.output_folder = self.output_folder.replace(self.__class__.__name__, self.__class__.__name__+'-'+self.sym_range)
        self.output_folder_base = str(Path(self.output_folder).parent)
        self.print_to_log_file(f"Using sym_range '{self.sym_range}' for nnUNetTrainerV2_SymPermute (set env var SYM_PERMUTE_RANGE to use 'full' or 'six-neighbourhood-only').")

    def initialize_network(self):
        self.base_num_features = 24  # otherwise we run out of VRAM
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
        self.net_num_pool_op_kernel_sizes = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]])
        self.net_pool_per_axis = np.array([4, 4, 4])

        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    2, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)

        for name, module in self.network.named_modules():
            if isinstance(module, nn.Conv3d):
                # Get current layer
                bn = get_layer(self.network, name)
                # Create new gn layer
                if(bn.kernel_size[0] == 3):
                    gn = PermuteConv3D(bn.in_channels,bn.out_channels,bn.kernel_size, stride=bn.stride, padding=bn.padding, dilation=bn.dilation, groups=bn.groups, bias=bn.bias, sym_range=self.sym_range)
                    # Assign gn
                    self.print_to_log_file(f"Swapping 3 {bn} with {gn}")
                    set_layer(self.network, name, gn)
            if isinstance(module, nn.ConvTranspose3d):
                # Get current layer
                bn = get_layer(self.network, name)
                # Create new gn layer
                gn = nn.Sequential(nn.Conv3d(bn.in_channels,bn.out_channels,1,bias=False),nn.Upsample(scale_factor=bn.kernel_size,mode='trilinear'))
                NEWLINE = '\n'
                self.print_to_log_file(f"Swapping {bn} with {str(gn).replace(NEWLINE, '')}")
                set_layer(self.network, name, gn)

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper