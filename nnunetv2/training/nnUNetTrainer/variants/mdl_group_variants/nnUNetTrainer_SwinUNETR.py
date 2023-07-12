import torch
from torch import nn
from monai.networks.nets import SwinUNETR

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

from nnunetv2.training.nnUNetTrainer.variants.mdl_group_variants.nnUNetTrainer_ClearML import nnUNetTrainer_ClearML


class SwinUNETR_Local(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = SwinUNETR(
            img_size=(128, 128, 128),
            in_channels=1,
            out_channels=12,
            feature_size=48,
            use_checkpoint=True
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

        self.initial_lr = args.custom_initial_lr

    def build_network_architecture(
            self,
            plans_manager: PlansManager,
            dataset_json,
            configuration_manager: ConfigurationManager,
            num_input_channels,
            enable_deep_supervision: bool = True
    ) -> nn.Module:

        network = SwinUNETR_Local()

        if self.settings.custom_pretrained_weights:
            self.restore_ckpt(network, self.settings.custom_pretrained_weights)

        return network

    def set_deep_supervision_enabled(self, enabled: bool):
        self.network.tuple_output = enabled

    def restore_ckpt(self, network, ckpt_path=''):

        # Load weights of SSL pre-trained encoder
        state_dict = torch.load(ckpt_path)["state_dict"]

        # Fix potential internal naming mismatches
        for key in list(state_dict.keys()):
            if 'module.' in key:
                state_dict[key.replace("module.", "model.swinViT.")] = state_dict.pop(key)

            elif 'swinViT.' in key:
                state_dict[key.replace("swinViT.", "model.swinViT.")] = state_dict.pop(key)

        for key in list(state_dict.keys()):
            if 'fc' in key:
                state_dict[key.replace("fc", "linear")] = state_dict.pop(key)

        # Load model weights with parameter`strict` set to False:
        #   encoder weights will be loaded (Swin-ViT, SSL pre-trained),
        #   decoder weights will remain untouched (CNN UNet decoder).
        message = network.load_state_dict(state_dict, strict=False)

        print(f'missing_keys: {message.missing_keys}')
        print(f'unexpected_keys: {message.unexpected_keys}')
