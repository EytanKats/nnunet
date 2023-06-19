import os
import sys
import shlex

from meidic_vtach_utils.run_on_recommended_cuda import get_cuda_environ_vars as get_vars
os.environ.update(get_vars('*'))

os.environ['nnUNet_raw'] = "/share/data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw"
os.environ['nnUNet_preprocessed'] = "/share/data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_preprocessed"
os.environ['nnUNet_results'] = "/share/data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results"

DO_DEBUG = True

if DO_DEBUG:
    os.environ['NNUNET_DEBUG_FLAG'] = "1"
else:
    if 'NNUNET_DEBUG_FLAG' in os.environ: del os.environ['NNUNET_DEBUG_FLAG']

from nnunetv2.run.run_training import run_training_entry as nnunet_run_training_main

# TRAIN_COMMAND = "nnUNet_train 2d nnUNetTrainerV2_DeepSTAPLE 651 0"
# TRAIN_COMMAND = "nnUNet_train 3d_fullres nnUNetTrainerV2_XEdgeConvMax 999 all"
# os.environ['SYM_PERMUTE_RANGE'] = 'six-neighbourhood-only'
# os.environ['NNUNET_RFA_STRENGTH'] = "0.7"
# TRAIN_COMMAND = "nnUNetv2_train -tr nnUNetTrainer_MIC 803 3d_fullres 0 --num_epochs 150 --disable_mirroring --mic_dropout 0.7 --mic_num_patches 16"
TRAIN_COMMAND = "nnUNetv2_train -tr nnUNetTrainer 505 3d_fullres 0 --num_epochs 300 --disable_mirroring --c"

sys.argv = shlex.split(TRAIN_COMMAND)
nnunet_run_training_main()