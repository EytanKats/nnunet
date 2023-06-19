import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from simple_converge.mlops.MLOpsTask import MLOpsTask


class nnUNetTrainer_ClearML(nnUNetTrainer):

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

        self.settings = args

        mlops_settings = {
            'use_mlops': self.settings.use_mlops,
            'project_name': self.settings.mlops_project_name,
            'task_name': self.settings.mlops_task_name,
            'task_type': 'training',
            'tags': self.settings.mlops_tags,
            'connect_arg_parser': False,
            'connect_frameworks': False,
            'resource_monitoring': True,
            'connect_streams': True
        }

        self.mlops_task = MLOpsTask(settings=mlops_settings)

    def on_epoch_end(self):
        super().on_epoch_end()

        epoch_time = self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1]
        self.mlops_task.log_scalar_to_mlops_server(f'epoch_time', f'epoch_time', epoch_time, self.current_epoch)

        loss_train = self.logger.my_fantastic_logging['train_losses'][-1]
        self.mlops_task.log_scalar_to_mlops_server(f'loss', f'train', loss_train, self.current_epoch)

        loss_val = self.logger.my_fantastic_logging['val_losses'][-1]
        self.mlops_task.log_scalar_to_mlops_server(f'loss', f'val', loss_val, self.current_epoch)

        ema_fg_dice = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
        self.mlops_task.log_scalar_to_mlops_server(f'fg_dice', f'ema_fg_dice', ema_fg_dice, self.current_epoch)

        labels = self.dataset_json['labels']
        pseudo_dice = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]
        for key, value in labels.items():

            # skip background
            if value == 0:
                continue

            self.mlops_task.log_scalar_to_mlops_server('pseudo_dice', key, pseudo_dice[value - 1], self.current_epoch)






