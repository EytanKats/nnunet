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
    # pass
else:
    if 'NNUNET_DEBUG_FLAG' in os.environ: del os.environ['NNUNET_DEBUG_FLAG']

from nnunetv2.inference.predict_from_raw_data import predict_entry_point as main

PREDICT_STR = \
    "nnUNetv2_predict " \
    "-i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/imagesTs " \
    "-o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-AMOS_nnUNetTrainer_MIND " \
    "-d 801 " \
    "-tr nnUNetTrainer_MIND " \
    "-c 3d_fullres " \
    "-f 0 "

# TODO remove following
# os.environ['NNUNET_RFA_STRENGTH'] = "0.0"
# PREDICT_STR = "nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-AMOS_nnUNetTrainer_ParametrizedRFA-0.0 -d 801 -tr nnUNetTrainer_ParametrizedRFA-0.7 -c 3d_fullres -f 0"
PREDICT_STR = "nnUNetv2_train -tr nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_MIND_MIC-np16_dr0.7_dcTrue -d 803 -tr nnUNetTrainer_MIND_MIC-np16_dr0.7_dcTrue -c 3d_fullres -f 0 --mind_mic_dropout 0.7 --mind_mic_num_patches 16 --mind_mic_different_per_channel --disable_tta"
print(PREDICT_STR)
sys.argv = shlex.split(PREDICT_STR)
main()