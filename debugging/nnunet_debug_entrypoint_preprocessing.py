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

from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry as main

PREPROCESS_COMMAND = 'nnUNetv2_plan_and_preprocess --verify_dataset_integrity -d 800 --clean -c 3f_fullres'
sys.argv = shlex.split(PREPROCESS_COMMAND)
main()