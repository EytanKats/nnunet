import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

import argparse
parser = argparse.ArgumentParser(prog='nnUNetTrainer_ParametrizedRFA')
parser.add_argument('--rfa_strength', type=float, required=True)


def rfa(x, strength=0.5):
    if x.dim() == 5:
        mode = '3d'
    elif x.dim() == 4:
        mode = '2d'
    else:
        raise(ValueError())

    with torch.no_grad():
        device = x.device

        if mode == '2d':
            B,_,H,W = x.size()
            field = \
                F.avg_pool2d(
                    F.avg_pool2d(torch.randn(B, 2, H//4, W//4).to(device), 7, stride=1, padding=3),
                    7, stride=1, padding=3
                )
            full_field = F.interpolate(field, scale_factor=4, mode='bilinear')

        elif mode == '3d':
            B,_,D,H,W = x.size()
            field = \
                F.avg_pool3d(
                    F.avg_pool3d(torch.randn(B, 2, D//4, H//4, W//4).to(device), 7, stride=1, padding=3),
                    7, stride=1, padding=3
                )
            full_field = F.interpolate(field, scale_factor=4, mode='trilinear')
        else:
            raise ValueError()

        # strenght of random field augmentation [0.0...1.0]
        y = x * ((1.-strength) + full_field[:,:1]) \
            + strength * ((x.max() - x.min()) / (full_field.max() - full_field.min())) * full_field[:,1:]
        y = y * (x.std() / y.std())

    return y



class nnUNetTrainer_ParametrizedRFA(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """RFA nnUNet"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        args = parser.parse_known_args()[0]
        self.rfa_strength = args.rfa_strength
        self.output_folder = self.output_folder.replace(self.__class__.__name__, self.__class__.__name__+'-'+str(self.rfa_strength))
        self.output_folder_base = self.output_folder_base.replace(self.__class__.__name__, self.__class__.__name__+'-'+str(self.rfa_strength))
        self.log_file = self.log_file.replace(self.__class__.__name__, self.__class__.__name__+'-'+str(self.rfa_strength))
        self.print_to_log_file(f"Random field augmentation strength is set to {self.rfa_strength}")


    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        network = get_network_from_plans(plans_manager, dataset_json, configuration_manager,
            num_input_channels, deep_supervision=enable_deep_supervision)

        def rfa_hook(module, input):
            input = input[0]
            if '--disable_tta' in sys.argv:
                pass
            else:
                args = parser.parse_known_args()[0]
                input = rfa(input, args.rfa_strength)
            return input

        network.register_forward_pre_hook(rfa_hook)

        return network