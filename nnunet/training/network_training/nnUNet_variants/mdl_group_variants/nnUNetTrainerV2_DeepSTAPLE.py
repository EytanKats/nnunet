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


from collections import OrderedDict
import itertools
from typing import Tuple
from time import time

import numpy as np
import torch
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation

from nnunet.training.loss_functions.dp_loss import DPLoss, MultipleOutputLoss2_FlexibleArgs
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.dataloading.dataset_loading import load_dataset, unpack_dataset
from nnunet.training.dataloading.dataset_loading_named_samples import  NamedSampleDataLoader3D, NamedSampleDataLoader2D
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *

class DynamicEmbedding(torch.nn.Module):

    def __init__(self, num_classes, init_value=0.0):
        super().__init__()
        self.torch_embedding = torch.nn.Embedding(num_embeddings=0, embedding_dim=num_classes, sparse=True)
        self.num_classes = num_classes
        self._map = {}
        self.init_value = init_value

    def insert_nonexisting(self, key_list):
        for key in key_list:
            if not key in self._map:
                last_len = len(self)
                expanded_embedding = torch.nn.Embedding(num_embeddings=last_len+1, embedding_dim=self.num_classes, sparse=True)
                expanded_embedding.to(device=self.torch_embedding.weight.device)
                expanded_embedding.weight.data[:-1] = self.torch_embedding.weight.data
                self.torch_embedding = expanded_embedding
                self.torch_embedding.weight.data[-1] = self.init_value
                self._map[key] = last_len

    def __call__(self, key_list):
        assert isinstance(key_list, list)

        self.insert_nonexisting(key_list)
        ids = []
        for key in key_list:
            ids.append(self._map[key])
        ids = torch.tensor(ids, device=self.torch_embedding.weight.device)

        return self.torch_embedding(ids)

    def __len__(self):
        return self.torch_embedding.num_embeddings

class nnUNetTrainerV2_DeepSTAPLE(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

        self.use_progress_bar = True

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = DPLoss(threeD=self.threeD, num_classes=self.num_classes)
            self.loss = MultipleOutputLoss2_FlexibleArgs(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    if 'NNUNET_DEBUG_FLAG' in os.environ:
                        unpack_dataset(self.folder_with_preprocessed_data, threads=1)
                    else:
                        unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
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
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
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
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        # Add embedding use self.num_classes, dynamic increase: https://discuss.pytorch.org/t/expand-an-existing-embedding-and-linear-layer-nan-loss-value/55670/2
        self.network.add_module('embedding', DynamicEmbedding(self.num_classes, 0.0))

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"

        self.network_parameters = [param_group for name, param_group in self.network.named_parameters() if not 'embedding' in name]
        self.optimizer = torch.optim.AdamW(self.network_parameters, lr=self.initial_lr)
        self.optimizer_dp = torch.optim.SparseAdam(list(self.network.embedding.torch_embedding.parameters()), lr=self.initial_lr*2, betas=(0.9, 0.999), eps=1e-08)
        self.lr_scheduler = None

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        bare_weight = self.network.embedding(data_dict['case_ids'])
        self.optimizer_dp.param_groups[0]['params'] = list(self.network.embedding.torch_embedding.parameters())

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()
        self.optimizer_dp.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target, bare_weight)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network_parameters, 12, error_if_nonfinite=False)

                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.step(self.optimizer_dp)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target, bare_weight)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network_parameters, 12, error_if_nonfinite=False)
                self.optimizer.step()
                self.optimizer_dp.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        mean_weight_string = np.array2string(self.network.embedding.torch_embedding.weight[:,1:].mean(dim=0).detach().cpu().numpy(),precision=2, floatmode='fixed', sign='+')
        self.print_to_log_file(f"Foreground mean data parameter values: {mean_weight_string}")
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = NamedSampleDataDataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = NamedSampleDataDataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:
            dl_tr = NamedSampleDataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = NamedSampleDataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time()
        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler,
                                                     'state_dict'):  # not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            # WTF is this!?
            # for key in lr_sched_state_dct.keys():
            #    lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
            optimizer_dp_state_dict = {
                'state': self.optimizer_dp.state,
                'param_groups': self.optimizer_dp.param_groups,
            }
        else:
            optimizer_state_dict = None
            optimizer_dp_state_dict = None

        self.print_to_log_file("saving checkpoint...")
        save_this = {
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                           self.all_val_eval_metrics),
            'best_stuff' : (self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA),
            'dynamic_embedding_map' : self.network.embedding._map,
            'optimizer_dp_state_dict' : optimizer_dp_state_dict}

        if self.amp_grad_scaler is not None:
            save_this['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()

        torch.save(save_this, fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time() - start_time))



    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        # Prepare embedding
        embedding_map = checkpoint['dynamic_embedding_map']
        case_ids =  list(embedding_map.keys())
        self.network.embedding(case_ids)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if train:
                if 'amp_grad_scaler' in checkpoint.keys():
                    self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        self.network.load_state_dict(new_state_dict)
        self.epoch = checkpoint['epoch']
        if train:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint[
            'plot_stuff']

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = checkpoint[
                'best_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]

        self._maybe_init_amp()