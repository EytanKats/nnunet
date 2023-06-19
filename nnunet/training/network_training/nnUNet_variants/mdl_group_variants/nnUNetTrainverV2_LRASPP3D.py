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


import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params, \
    default_2D_augmentation_params, get_patch_size
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import torchvision.models




from torch import nn, Tensor
from torch.nn import functional as F
from typing import Dict
from collections import OrderedDict

class LRASPP(nn.Module):
    """
    Implements a Lite R-ASPP Network for semantic segmentation from
    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "high" for the high level feature map and "low" for the low level feature map.
        low_channels (int): the number of channels of the low level features.
        high_channels (int): the number of channels of the high level features.
        num_classes (int): number of output classes of the model (including the background).
        inter_channels (int, optional): the number of channels for intermediate computations.
    """

    def __init__(
        self,
        backbone: nn.Module,
        low_channels: int,
        high_channels: int,
        num_classes: int,
        inter_channels: int = 128
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = LRASPPHead(low_channels, high_channels, num_classes, inter_channels)

    def forward(self, input: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(input)
        out = self.classifier(features)
        out = F.interpolate(out, size=(int(input.shape[2]/1.5),int(input.shape[3]/1.5),int(input.shape[4]/1.5)),\
                            mode='trilinear', align_corners=False)

        #result = OrderedDict()
        #result["out"] = out

        return out


class LRASPPHead(nn.Module):

    def __init__(
        self,
        low_channels: int,
        high_channels: int,
        num_classes: int,
        inter_channels: int
    ) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv3d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv3d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv3d(inter_channels, num_classes, 1)

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        low = input["low"]
        high = input["high"]

        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-3:], mode='trilinear', align_corners=False)

        return self.low_classifier(low) + self.high_classifier(x)

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


class LRASPP3d(SegmentationNetwork):
    def __init__(self, in_channels=2, out_channels=32, num_classes=2, deep_supervision=False):  # or out_channels = 16/64
        super(LRASPP3d, self).__init__()
        #weightInitializer=InitWeights_He(1e-2),
        self.do_ds = False
        self.num_classes = num_classes

        self.backbone = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=False,
            num_classes=self.num_classes).backbone

        self.model = LRASPP(self.backbone,40,960,self.num_classes,128)

        self.apply(InitWeights_He(1e-2))

    def forward(self, x):
        input = F.interpolate(x,scale_factor=1.5,mode='trilinear',align_corners=False)

        output = self.model(input)#self.model.classifier(self.model.backbone(input))

        #x = F.upsample_bilinear(output,scale_factor=2)
        return output#[0]

    def compute_approx_vram_consumption(self):
        return 1715000000



class nnUNetTrainerV2_LRASPP3D(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

    def setup_DA_params(self):
        """
        we leave out the creation of self.deep_supervision_scales, so it remains None
        :return:
        """
        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

    def initialize(self, training=True, force_load_plans=False):
        """
        removed deep supervision
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                assert self.deep_supervision_scales is None
                self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val,
                                                                    self.data_aug_params[
                                                                        'patch_size_for_spatialtransform'],
                                                                    self.data_aug_params,
                                                                    deep_supervision_scales=self.deep_supervision_scales,
                                                                    classes=None,
                                                                    pin_memory=self.pin_memory)

                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        changed deep supervision to False
        :return:
        """
        if self.threeD:
            self.conv_op = nn.Conv3d
            self.dropout_op = nn.Dropout3d
            self.norm_op = nn.InstanceNorm3d

        else:
            self.conv_op = nn.Conv2d
            self.dropout_op = nn.Dropout2d
            self.norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        self.network = LRASPP3d(self.num_input_channels, self.base_num_features, self.num_classes)
        self.network.conv_op = self.conv_op
        self.network.dropout_op = self.dropout_op
        self.network.norm_op = self.norm_op

        count = 0; count2 = 0
        for name, module in self.network.named_modules():
            if isinstance(module, nn.Conv2d):
                before = get_layer(self.network, name)
                kernel_size = tuple((list(before.kernel_size)*2)[:3])
                stride = tuple((list(before.stride)*2)[:3])
                padding = tuple((list(before.padding)*2)[:3])
                dilation = tuple((list(before.dilation)*2)[:3])
                in_channels = before.in_channels
                if(in_channels==3):
                    before.in_channels = 1
                after = nn.Conv3d(before.in_channels,before.out_channels,kernel_size,stride=stride,\
                                  padding=padding,dilation=dilation,groups=before.groups)
                set_layer(self.network, name, after); count += 1


            if isinstance(module, nn.BatchNorm2d):
                before = get_layer(self.network, name)
                after = nn.InstanceNorm3d(before.num_features)#,affine=before.affine)
                set_layer(self.network, name, after); count2 += 1
            if isinstance(module, nn.AdaptiveAvgPool2d):
                before = get_layer(self.network, name)
                after = nn.AdaptiveAvgPool3d(before.output_size)
                set_layer(self.network, name, after); count2 += 1
            if isinstance(module, nn.Hardswish):
                before = get_layer(self.network, name)
                after = nn.LeakyReLU(negative_slope=1e-2,inplace=True)
                set_layer(self.network, name, after); count2 += 1


        print(count,'# Conv2d > Conv3d','and',count2,'#batchnorm etc')

        #countParameters(network)





        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def run_online_evaluation(self, output, target):
        return nnUNetTrainer.run_online_evaluation(self, output, target)