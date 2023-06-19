import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

parser = argparse.ArgumentParser(prog='nnUNetTrainer_MIC')
parser.add_argument('--mic_dropout', type=float, required=True)
parser.add_argument('--mic_num_patches', type=int, required=True)
parser.add_argument('--mic_different_per_channel', action=argparse.BooleanOptionalAction)

def apply_mic(input, num_patches=16, dropout=0.7, different_per_channel=False):
    B, C, *_ = input.shape
    if not different_per_channel:
        C = 1

    mask = (torch.rand((B, C) +(num_patches,)*(input.dim()-2), device=input.device) > dropout).to(input.dtype)
    mask = F.interpolate(mask, size=input.shape[2:], mode='nearest')

    input = input * mask
    return input



class nnUNetTrainer_MIC(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """MIC nnUNet"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        arg_string = f"np{args.mic_num_patches}_dr{args.mic_dropout}_dc{args.mic_different_per_channel}"
        self.output_folder = self.output_folder.replace(self.__class__.__name__, self.__class__.__name__+'-'+arg_string)
        self.output_folder_base = self.output_folder_base.replace(self.__class__.__name__, self.__class__.__name__+'-'+arg_string)
        self.log_file = self.log_file.replace(self.__class__.__name__, self.__class__.__name__+'-'+arg_string)
        self.print_to_log_file(f"Using MIC trainer with: {arg_string}")

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        network = get_network_from_plans(plans_manager, dataset_json, configuration_manager,
            num_input_channels, deep_supervision=enable_deep_supervision)

        def mic_hook(module, input):
            input = input[0]

            if '--disable_tta' in sys.argv:
                pass
            else:
                args = parser.parse_known_args()[0]
                input = apply_mic(
                    input,
                    num_patches=args.mic_num_patches,
                    dropout=args.mic_dropout,
                    different_per_channel=args.mic_different_per_channel)

            return input

        network.register_forward_pre_hook(mic_hook)

        return network