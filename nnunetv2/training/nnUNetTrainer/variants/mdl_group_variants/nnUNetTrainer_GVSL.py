import torch
from torch import nn
from Downstream.models_finetune.UNet_GVSL import UNet3D_GVSL

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.training.nnUNetTrainer.variants.mdl_group_variants.nnUNetTrainer_ClearML import nnUNetTrainer_ClearML


class GVSL_Local(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = UNet3D_GVSL(
            n_classes=13,
            # pretrained_weights=None)
            pretrained_weights='/share/data_supergrover3/kats/experiments/label/gvsl/nako_1000_pretraining_plainunet/GVSL_epoch_20.pth')

        self.tuple_output = True

    def forward(self, x):

        x = self.model(x)

        if self.tuple_output:
            return (x,)
        else:
            return x


class nnUNetTrainer_trying_things(nnUNetTrainer_ClearML):

    def __init__(
            self,
            args,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device('cuda')
    ):
        super().__init__(args, plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.initial_lr = args.custom_initial_lr

    def build_network_architecture(
            self,
            plans_manager: PlansManager,
            dataset_json,
            configuration_manager: ConfigurationManager,
            num_input_channels,
            enable_deep_supervision: bool = True
    ) -> nn.Module:

        network = GVSL_Local()
        # network = UNet3D_GVSL(
        #     n_classes=13,
        #     # pretrained_weights=None)
        #     pretrained_weights='/share/data_supergrover3/kats/experiments/label/gvsl/nako_1000_pretraining_plainunet/GVSL_epoch_120.pth')
        return network

    def set_deep_supervision_enabled(self, enabled: bool):
        # self.network.unet.deep_supervision = enabled
        self.network.tuple_output = enabled
