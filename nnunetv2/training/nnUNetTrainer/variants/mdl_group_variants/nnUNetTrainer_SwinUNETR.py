import torch
from torch import nn
from monai.networks.nets import SwinUNETR

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

from nnunetv2.training.nnUNetTrainer.variants.mdl_group_variants.nnUNetTrainer_ClearML import nnUNetTrainer_ClearML


class SwinUNETR_Local(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = SwinUNETR(
            img_size=(64, 192, 192),
            in_channels=1,
            out_channels=14,
            feature_size=48
        )

        self.tuple_output = True

    def forward(self, x):

        x = self.model(x)

        if self.tuple_output:
            return (x,)
        else:
            return x


class nnUNetTrainer_SwinUNETR(nnUNetTrainer_ClearML):

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

    @staticmethod
    def build_network_architecture(
            plans_manager: PlansManager,
            dataset_json,
            configuration_manager: ConfigurationManager,
            num_input_channels,
            enable_deep_supervision: bool = True
    ) -> nn.Module:

        network = SwinUNETR_Local()

        return network

    def set_deep_supervision_enabled(self, enabled: bool):
        self.network.tuple_output = enabled

