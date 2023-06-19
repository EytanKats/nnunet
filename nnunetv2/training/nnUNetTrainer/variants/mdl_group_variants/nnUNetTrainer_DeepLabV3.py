import os
import sys
import shutil
import argparse
from typing import List, Dict
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision.models
from torchvision.models._utils import IntermediateLayerGetter

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from batchgenerators.utilities.file_and_folder_operations import join, save_json, maybe_mkdir_p
from nnunetv2.training.dataloading.utils import unpack_dataset
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

from nnunetv2.training.nnUNetTrainer.variants.mdl_group_variants.deeplab_v3_3d import DeepLabV3_3D

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv3d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, num_classes, 1),
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv3d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-3:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv3d(in_channels, out_channels, 1, bias=False), nn.BatchNorm3d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv3d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


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


def replace2d_by_3d(network):
    count = 0; count2 = 0
    for name, module in network.named_modules():
        if isinstance(module, nn.Conv2d):
            before = get_layer(network, name)
            kernel_size = tuple((list(before.kernel_size)*2)[:3])
            stride = tuple((list(before.stride)*2)[:3])
            padding = tuple((list(before.padding)*2)[:3])
            dilation = tuple((list(before.dilation)*2)[:3])
            in_channels = before.in_channels
            if(in_channels==3):
                before.in_channels = 1
                stride=1
            after = nn.Conv3d(before.in_channels,before.out_channels,kernel_size,stride=stride,\
                              padding=padding,dilation=dilation,groups=before.groups)
            set_layer(network, name, after); count += 1

        if isinstance(module, nn.BatchNorm2d):
            before = get_layer(network, name)
            after = nn.InstanceNorm3d(before.num_features)#,affine=before.affine)
            set_layer(network, name, after); count2 += 1
        if isinstance(module, nn.AdaptiveAvgPool2d):
            before = get_layer(network, name)
            after = nn.AdaptiveAvgPool3d(before.output_size)
            set_layer(network, name, after); count2 += 1
        if isinstance(module, nn.MaxPool2d):
            before = get_layer(network, name)
            after = nn.MaxPool3d(2)
            set_layer(network, name, after); count2 += 1
        if isinstance(module, nn.Hardswish):
            before = get_layer(network, name)
            after = nn.LeakyReLU(negative_slope=1e-2,inplace=True)
            set_layer(network, name, after); count2 += 1
    print(count,'# Conv2d > Conv3d','and',count2,'#batchnorm etc')
    return network


class CheckpointingGradWrapper(nn.Module):
    # see https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, ctx, dummy_arg=None):
        assert dummy_arg.requires_grad
        ctx = self.module(ctx)
        return ctx


class DeepLabV3(nn.Module):
    def __init__(self, in_channels=2, out_channels=32, num_classes=2, deep_supervision=False, use_checkpointing=True):  # or out_channels = 16/64
        super(DeepLabV3, self).__init__()
        #weightInitializer=InitWeights_He(1e-2),
        # self.do_ds = False

        self.num_classes = num_classes
        return_layers = {"layer3": "out"}
        resnet = replace2d_by_3d(torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False,num_classes=num_classes).backbone)

        self.backbone = IntermediateLayerGetter(resnet, return_layers=return_layers)

        self.classifier = DeepLabHead(1024,num_classes)
        self.apply(InitWeights_He(1e-2))

        self.use_checkpointing = use_checkpointing
        self.grad_dummy = torch.tensor(1., requires_grad=True)

    def forward(self, x):
        input = x
        # input = F.interpolate(x,scale_factor=1.5,mode='trilinear',align_corners=False)
        input_shape = x.shape[-3:]
        # contract: features is a dict of tensors
        if self.use_checkpointing:
            features = checkpoint(
                CheckpointingGradWrapper(self.backbone), input, self.grad_dummy
            )
        else:
            features = self.backbone(input)

        x = features["out"]
        if self.use_checkpointing:
            x = checkpoint(
                CheckpointingGradWrapper(self.classifier), x, self.grad_dummy
            )
        else:
            x = self.classifier(x)

        x = F.interpolate(x, size=input_shape, mode="trilinear", align_corners=False)

        return (x,)


class nnUNetTrainer_DeepLabV3(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """MIC nnUNet"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        num_classes = plans_manager.get_label_manager(dataset_json).num_segmentation_heads
        # network = DeepLabV3(in_channels=num_input_channels, out_channels=32, num_classes=num_classes, deep_supervision=False)

        network = DeepLabV3_3D(num_classes=num_classes, input_channels=num_input_channels, resnet='resnet34_os16')

        return network


