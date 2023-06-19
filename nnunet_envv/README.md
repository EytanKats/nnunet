# Run nnunet preprocessing

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_plan_and_preprocess --verify_dataset_integrity -d 505 --clean'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -t 556'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -t 559'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -t 560'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -t 561'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -t 562'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -t 563'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -t 650'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -t 651'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -t 652'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -t 653'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -t 654'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -t 655'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -t 656'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -t 657'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -t 658'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity --no-check-geometry -t 659'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -t 700'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_plan_and_preprocess --verify_dataset_integrity -d 800 --clean'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_plan_and_preprocess --verify_dataset_integrity -d 801 --clean'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_plan_and_preprocess --verify_dataset_integrity -d 802 --clean'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_plan_and_preprocess --verify_dataset_integrity -d 803 --clean'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_plan_and_preprocess --verify_dataset_integrity -d 804 --clean'

# nnunetv1
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -d 803'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -d 804'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_plan_and_preprocess --verify_dataset_integrity -t 998'

# Run nnunet training

# 202

source ./set_envs.sh && SYM_PERMUTE_RANGE=full run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_SymPermute 202 0 --epochs 150'

source ./set_envs.sh && SYM_PERMUTE_RANGE=six-neighbourhood-only run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_SymPermute 202 0 --epochs 150'

# 505
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer 505 3d_fullres 0 --num_epochs 300 --disable_mirroring'
source ./set_envs.sh && NNUNET_DEBUG_FLAG=1 run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer 505 3d_fullres 0 --num_epochs 300 --disable_mirroring --c'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer 505 3d_fullres 1 --num_epochs 300 --disable_mirroring'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer 505 3d_fullres 2 --num_epochs 300 --disable_mirroring'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer 505 3d_fullres 3 --num_epochs 300 --disable_mirroring'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer 505 3d_fullres 4 --num_epochs 300 --disable_mirroring'

# 555
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_insaneDA 555 all'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2 556 all'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2 557 all'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2 558 all'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2 559 all'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_insaneDA 560 all'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_insaneDA 561 all'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_insaneDA 562 all'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2 563 all'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2 650 all'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_RandomFieldAugmentation 650 all'

# 651
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2 651 all'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_RandomFieldAugmentation 651 all'
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.2 run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_ParametrizedRandomFieldAugmentation 651 all --epochs 150'
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.6 run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_ParametrizedRandomFieldAugmentation06 651 all --epochs 150'
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_ParametrizedRandomFieldAugmentation07 651 all --epochs 150'
source ./set_envs.sh && NNUNET_RFA_STRENGTH=1.0 run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_ParametrizedRandomFieldAugmentation10 651 all --epochs 150'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_insaneDA 651 all'

source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_ParametrizedRandomFieldAugmentation07 651 0 --epochs 1000'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_DeepSTAPLE 651 all'

# 652
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2 652 all'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_RandomFieldAugmentation 652 all'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2 653 all'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_RandomFieldAugmentation 653 all'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_RandomFieldAugmentation 654 0'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_RandomFieldAugmentation 655 all --epochs 250'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_XEdgeConvMax 655 all --epochs 250'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2 655 all --epochs 250'

# 656
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_RandomFieldAugmentation 656 all --epochs 250'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_XEdgeConvMax 656 all --epochs 250'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2 656 all --epochs 250'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_XEdgeConvMax 656 0 --epochs 150'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_XEdgeConvMax 656 1 --epochs 150'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_XEdgeConvMax 656 2 --epochs 150'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_XEdgeConvMax 656 3 --epochs 150'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_XEdgeConvMax 656 4 --epochs 150'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_RandomFieldAugmentationXEdgeConvMax 656 all --epochs 250'

# 657
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_RandomFieldAugmentation 657 all --epochs 250'

# 658
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_ParametrizedRandomFieldAugmentation 658 0 --epochs 150'
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_ParametrizedRandomFieldAugmentation 658 1 --epochs 150'
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_ParametrizedRandomFieldAugmentation 658 2 --epochs 150'
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_ParametrizedRandomFieldAugmentation 658 3 --epochs 150'
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_ParametrizedRandomFieldAugmentation 658 4 --epochs 150'

# 659
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_ParametrizedRandomFieldAugmentation 659 0 --epochs 150'
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_ParametrizedRandomFieldAugmentation 659 1 --epochs 150'
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_ParametrizedRandomFieldAugmentation 659 2 --epochs 150'
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_ParametrizedRandomFieldAugmentation 659 3 --epochs 150'
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_ParametrizedRandomFieldAugmentation 659 4 --epochs 150'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2 659 0 --epochs 150 --c'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2 659 1 --epochs 150'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2 659 2 --epochs 150'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2 659 3 --epochs 150'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2 659 4 --epochs 150'

# 700 RFA
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_RandomFieldAugmentation 700 0 --epochs 250'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_RandomFieldAugmentation 700 1 --epochs 250'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_RandomFieldAugmentation 700 2 --epochs 250'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_RandomFieldAugmentation 700 3 --epochs 250'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2_RandomFieldAugmentation 700 4 --epochs 250'

# 700 vanilla
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2 700 0 --epochs 250'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2 700 1 --epochs 250'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2 700 2 --epochs 250'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2 700 3 --epochs 250'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2 700 4 --epochs 250'

# 800
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer 800 3d_fullres 0 --num_epochs 150'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer 801 3d_fullres 0 --num_epochs 150'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_MIND 801 3d_fullres 0 --num_epochs 150'
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_ParametrizedRFA 801 3d_fullres 0 --num_epochs 150'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_GIN 801 3d_fullres 0 --num_epochs 150'

# 802
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer 802 3d_fullres 0 --num_epochs 150'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_MIND 802 3d_fullres 0 --num_epochs 150'
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_ParametrizedRFA 802 3d_fullres 0 --num_epochs 150'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_GIN 802 3d_fullres 0 --num_epochs 150'

# 803
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer 803 2d 0 --num_epochs 150 --disable_mirroring'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer 803 3d_fullres 0 --num_epochs 150 --disable_mirroring'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_insaneDAnoMirroring 803 0'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_MIND 803 3d_fullres 0 --num_epochs 150 --disable_mirroring'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_GIN 803 3d_fullres 0 --num_epochs 150 --disable_mirroring'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_GIN_MIND 803 3d_fullres 0 --num_epochs 150 --disable_mirroring'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_MIND_GIN 803 3d_fullres 0 --num_epochs 150 --disable_mirroring'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_MIND_MIC 803 3d_fullres 0 --num_epochs 150 --disable_mirroring --mind_mic_dropout 0.7 --mind_mic_num_patches 16 --no-mind_mic_different_per_channel'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_MIND_MIC 803 3d_fullres 0 --num_epochs 150 --disable_mirroring --mind_mic_dropout 0.7 --mind_mic_num_patches 16 --mind_mic_different_per_channel'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_MIC 803 3d_fullres 0 --num_epochs 150 --disable_mirroring --mic_dropout 0.7 --mic_num_patches 16'
dsource ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_MIC 803 3d_fullres 0 --num_epochs 150 --disable_mirroring --mic_dropout 0.3 --mic_num_patches 16'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainerBN 803 3d_fullres 0 --num_epochs 150 --disable_mirroring'

# 804
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer 804 2d 0 --num_epochs 150 --disable_mirroring'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer 804 3d_fullres 0 --num_epochs 150 --disable_mirroring'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 3d_fullres nnUNetTrainerV2_insaneDAnoMirroring 804 0'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_MIND 804 3d_fullres 0 --num_epochs 150 --disable_mirroring'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_GIN 804 3d_fullres 0 --num_epochs 150 --disable_mirroring'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_GIN_MIND 804 3d_fullres 0 --num_epochs 150 --disable_mirroring'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_MIND_GIN 804 3d_fullres 0 --num_epochs 150 --disable_mirroring'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_MIC 804 3d_fullres 0 --num_epochs 150 --disable_mirroring --mic_dropout 0.7 --mic_num_patches 16'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_MIC 804 3d_fullres 0 --num_epochs 150 --disable_mirroring --mic_dropout 0.3 --mic_num_patches 16'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainer_DeepLabV3 804 3d_fullres 0 --num_epochs 150 --disable_mirroring'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_train -tr nnUNetTrainerBN 804 3d_fullres 0 --num_epochs 150 --disable_mirroring'

# 998
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_train 2d nnUNetTrainerV2 700 4 --epochs 250'

# Postprocessing
source ./set_envs.sh && nnUNet_determine_postprocessing -m 2d -tr nnUNetTrainerV2 -t 700
source ./set_envs.sh && nnUNet_determine_postprocessing -m 2d -tr nnUNetTrainerV2_RandomFieldAugmentation -t 700

source ./set_envs.sh && nnUNet_determine_postprocessing -m 3d_fullres -tr nnUNetTrainerV2_ParametrizedRandomFieldAugmentation -t 658

source ./set_envs.sh && nnUNet_determine_postprocessing -m 2d -tr nnUNetTrainerV2_ParametrizedRandomFieldAugmentation -t 659

source ./set_envs.sh && nnUNet_determine_postprocessing -m 2d -tr nnUNetTrainerV2 -t 659

# Prediction
# Copy model latest model to final checkpoint model
# cp ./nnUNet_trained_models/nnUNet/3d_fullres/Task555_CM_consensus_insane_all/nnUNetTrainerV2_insaneDA__nnUNetPlansv2.1/all/model_latest.model ./nnUNet_trained_models/nnUNet/3d_fullres/Task555_CM_consensus_insane_all/nnUNetTrainerV2_insaneDA__nnUNetPlansv2.1/all/model_final_checkpoint.model
# Replace 'latest' with 'final_checkpoint'
# Predicting with remote directory paths in /share can cause hanging of prediction process

# 202
source ./set_envs.sh && SYM_PERMUTE_RANGE=full run_on_recommended_cuda --command 'nnUNet_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task202_AbdomenCTCT/permuted_imagesTs_out -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task202_AbdomenCTCT_permuted/nnUNetTrainerV2_SymPermute-full -t 202 -m 3d_fullres -f 0 -tr nnUNetTrainerV2_SymPermute-full'

source ./set_envs.sh && SYM_PERMUTE_RANGE=six-neighbourhood-only run_on_recommended_cuda --command 'nnUNet_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task202_AbdomenCTCT/permuted_imagesTs_out -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task202_AbdomenCTCT_permuted/nnUNetTrainerV2_SymPermute-six-neighbourhood-only -t 202 -m 3d_fullres -f 0 -tr nnUNetTrainerV2_SymPermute-six-neighbourhood-only'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task202_AbdomenCTCT/permuted_imagesTs_out -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task202_AbdomenCTCT_permuted/nnUNetTrainerV2_lraspp3d -t 202 -m 3d_fullres -f 0 -tr nnUNetTrainerV2_lraspp3d'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task202_AbdomenCTCT/permuted_imagesTs_in -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task202_AbdomenCTCT_normal/nnUNetTrainerV2_lraspp3d -t 202 -m 3d_fullres -f 0 -tr nnUNetTrainerV2_lraspp3d'


# 556
source ./nnunet_env/set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/tmp/400_convex_adam/val_images -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task556_CM_consensus_random_convex_adam -t 556 -m 3d_fullres -f all -tr nnUNetTrainerV2

source ./nnunet_env/set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/tmp/400_convex_adam/val_images -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task557_CM_consensus_dp_convex_adam -t 557 -m 3d_fullres -f all -tr nnUNetTrainerV2

source ./nnunet_env/set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/tmp/400_convex_adam/val_images -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task558_CM_consensus_staple_convex_adam -t 558 -m 3d_fullres -f all -tr nnUNetTrainerV2

source ./nnunet_env/set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/tmp/400_convex_adam/val_images -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task559_CM_consensus_all_convex_adam -t 559 -m 3d_fullres -f all -tr nnUNetTrainerV2'


source ./nnunet_env/set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /data_rechenknecht01_2/weihsbach/nnunet/tmp/400_deeds/val_images -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task561_CM_domain_adaptation_insane_moving_deeds -t 561 -m 3d_fullres -f all -tr nnUNetTrainerV2_insaneDA --num_threads_preprocessing 1'


source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/tmp/400_convex_adam/val_images -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task562_CM_domain_adaptation_insane_moving_convex_adam -t 562 -m 3d_fullres -f all -tr nnUNetTrainerV2_insaneDA


source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/tmp/400_convex_adam/val_images -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task563_CM_consensus_expert_convex_adam -t 563 -m 3d_fullres -f all -tr nnUNetTrainerV2


source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task650_MMWHS_MRI/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task650_MMWHS_MRI/nnUNetTrainerV2 -t 650 -m 2d -f all -tr nnUNetTrainerV2

source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task650_MMWHS_MRI/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task650_MMWHS_MRI/nnUNetTrainerV2_RandomFieldAugmentation -t 650 -m 2d -f all -tr nnUNetTrainerV2_RandomFieldAugmentation

# 651
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task651_MMWHS_MRI_HLA/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task651_MMWHS_MRI_HLA/nnUNetTrainerV2 -t 651 -m 2d -f all -tr nnUNetTrainerV2'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task651_MMWHS_MRI_HLA/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task651_MMWHS_MRI_HLA/nnUNetTrainerV2_RandomFieldAugmentation -t 651 -m 2d -f all -tr nnUNetTrainerV2_RandomFieldAugmentation'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task651_MMWHS_MRI_HLA/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task651_MMWHS_MRI_HLA/nnUNetTrainerV2_ParametrizedRandomFieldAugmentation -t 651 -m 2d -f all -tr nnUNetTrainerV2_ParametrizedRandomFieldAugmentation'

source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.0 run_on_recommended_cuda --command 'nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task651_MMWHS_MRI_HLA/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task651_MMWHS_MRI_HLA/nnUNetTrainerV2_ParametrizedRandomFieldAugmentation06 -t 651 -m 2d -f all -tr nnUNetTrainerV2_ParametrizedRandomFieldAugmentation06'

source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.0 run_on_recommended_cuda --command 'nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task651_MMWHS_MRI_HLA/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task651_MMWHS_MRI_HLA/nnUNetTrainerV2_ParametrizedRandomFieldAugmentation07 -t 651 -m 2d -f all -tr nnUNetTrainerV2_ParametrizedRandomFieldAugmentation07'

source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.0 run_on_recommended_cuda --command 'nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task651_MMWHS_MRI_HLA/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task651_MMWHS_MRI_HLA/nnUNetTrainerV2_ParametrizedRandomFieldAugmentation10 -t 651 -m 2d -f all -tr nnUNetTrainerV2_ParametrizedRandomFieldAugmentation10'


source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task651_MMWHS_MRI_HLA/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task651_MMWHS_MRI_HLA/nnUNetTrainerV2_insaneDA -t 651 -m 2d -f all -tr nnUNetTrainerV2_insaneDA -chk model_latest'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task651_MMWHS_MRI_HLA/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task651_MMWHS_MRI_HLA/nnUNetTrainerV2_DeepSTAPLE -t 651 -m 2d -f all -tr nnUNetTrainerV2_DeepSTAPLE -chk model_best'

# 652
source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task652_MMWHS_MRI_SA/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task652_MMWHS_MRI_SA/nnUNetTrainerV2 -t 652 -m 2d -f all -tr nnUNetTrainerV2

source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task652_MMWHS_MRI_SA/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task652_MMWHS_MRI_SA/nnUNetTrainerV2_RandomFieldAugmentation -t 652 -m 2d -f all -tr nnUNetTrainerV2_RandomFieldAugmentation

# 653
source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task653_ACDC_SA/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task653_ACDC_SA/nnUNetTrainerV2 -t 653 -m 2d -f all -tr nnUNetTrainerV2

source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task653_ACDC_SA/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task653_ACDC_SA/nnUNetTrainerV2_RandomFieldAugmentation -t 653 -m 2d -f all -tr nnUNetTrainerV2_RandomFieldAugmentation

# 654
source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task654_MMWHS_MRI_HLA_PACKED/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task654_MMWHS_MRI_HLA_PACKED/nnUNetTrainerV2_RandomFieldAugmentation -t 654 -m 2d -f 0 -tr nnUNetTrainerV2_RandomFieldAugmentation

# 655
source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task655_MMWHS_COMMON_SPACE/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task655_MMWHS_COMMON_SPACE/nnUNetTrainerV2 -t 655 -m 3d_fullres -f all -tr nnUNetTrainerV2

source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task655_MMWHS_COMMON_SPACE/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task655_MMWHS_COMMON_SPACE/nnUNetTrainerV2_RandomFieldAugmentation -t 655 -m 3d_fullres -f all -tr nnUNetTrainerV2_RandomFieldAugmentation

source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task655_MMWHS_COMMON_SPACE/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task655_MMWHS_COMMON_SPACE/nnUNetTrainerV2_XEdgeConvMax -t 655 -m 3d_fullres -f all -tr nnUNetTrainerV2_XEdgeConvMax

# 656
source ./set_envs.sh && nnUNet_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task656_MMWHS_RESAMPLE_ONLY/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task656_MMWHS_RESAMPLE_ONLY/nnUNetTrainerV2 -t 656 -m 3d_fullres -f all -tr nnUNetTrainerV2

source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task656_MMWHS_RESAMPLE_ONLY/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task656_MMWHS_RESAMPLE_ONLY/nnUNetTrainerV2_RandomFieldAugmentation -t 656 -m 3d_fullres -f all -tr nnUNetTrainerV2_RandomFieldAugmentation

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task656_MMWHS_RESAMPLE_ONLY/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task656_MMWHS_RESAMPLE_ONLY/nnUNetTrainerV2_XEdgeConvMax -t 656 -m 3d_fullres -f all -tr nnUNetTrainerV2_XEdgeConvMax'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task656_MMWHS_RESAMPLE_ONLY/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task656_MMWHS_RESAMPLE_ONLY/nnUNetTrainerV2_RandomFieldAugmentationXEdgeConvMax -t 656 -m 3d_fullres -f all -tr nnUNetTrainerV2_RandomFieldAugmentationXEdgeConvMax'

source ./set_envs.sh && SYM_PERMUTE_RANGE=full run_on_recommended_cuda --command 'nnUNet_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task656_MMWHS_RESAMPLE_ONLY/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task656_MMWHS_RESAMPLE_ONLY/nnUNetTrainerV2_SymPermute-full -t 656 -m 3d_fullres -f all -tr nnUNetTrainerV2_SymPermute-full'

source ./set_envs.sh && SYM_PERMUTE_RANGE=six-neighbourhood-only run_on_recommended_cuda --command 'nnUNet_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task656_MMWHS_RESAMPLE_ONLY/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task656_MMWHS_RESAMPLE_ONLY/nnUNetTrainerV2_SymPermute-six-neighbourhood-only -t 656 -m 3d_fullres -f all -tr nnUNetTrainerV2_SymPermute-six-neighbourhood-only'

source ./set_envs.sh && SYM_PERMUTE_RANGE=six-neighbourhood-only run_on_recommended_cuda --command 'nnUNet_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task656_MMWHS_RESAMPLE_ONLY/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task656_MMWHS_RESAMPLE_ONLY/nnUNetTrainerV2_LRASPP3D -t 656 -m 3d_fullres -f all -tr nnUNetTrainerV2_LRASPP3D'


# 657
source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task657_MMWHS_ACDC_COMBINED_SA/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task657_MMWHS_ACDC_COMBINED_SA/nnUNetTrainerV2_RandomFieldAugmentation -t 657 -m 2d -f all -tr nnUNetTrainerV2_RandomFieldAugmentation

source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht03_2/students/kannath/ACDC/ACDC-Segmentation/split_fourdee_images -o /share/data_rechenknecht03_2/students/kannath/ACDC/ACDC-Segmentation/split_fourdee_segmentations -t 657 -m 2d -f all -tr nnUNetTrainerV2_RandomFieldAugmentation

# 658
source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task658_MMWHS_REGISTERED_LOWRES/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task658_MMWHS_REGISTERED_LOWRES/nnUNetTrainerV2_ParametrizedRandomFieldAugmentation -t 658 -m 3d_fullres -tr nnUNetTrainerV2_ParametrizedRandomFieldAugmentation

# 659
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task659_MMWHS_SLICES_HIRES/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task659_MMWHS_SLICES_HIRES/nnUNetTrainerV2_ParametrizedRandomFieldAugmentation -t 659 -m 2d -tr nnUNetTrainerV2_ParametrizedRandomFieldAugmentation

# 659
source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task659_MMWHS_SLICES_HIRES/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task659_MMWHS_SLICES_HIRES_MIUA/nnUNetTrainerV2 -t 659 -m 2d -tr nnUNetTrainerV2

# 700
source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task700_CMRxMotionTask2/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task700_CMRxMotionTask2/nnUNetTrainerV2 -t 700 -m 2d -f 0 1 2 3 4 -tr nnUNetTrainerV2

source ./set_envs.sh && nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task700_CMRxMotionTask2/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task700_CMRxMotionTask2/nnUNetTrainerV2_RandomFieldAugmentation -t 700 -m 2d -f 0 1 2 3 4 -tr nnUNetTrainerV2_RandomFieldAugmentation

# 801
# nnUNetTrainer
source ./set_envs.sh && nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-AMOS_nnUNetTrainer -d 801 -tr nnUNetTrainer -c 3d_fullres -f 0
source ./set_envs.sh && nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-BCV_nnUNetTrainer -d 801 -tr nnUNetTrainer -c 3d_fullres -f 0
# MIND trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-AMOS_nnUNetTrainer_MIND -d 801 -tr nnUNetTrainer_MIND -c 3d_fullres -f 0'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-BCV_nnUNetTrainer_MIND -d 801 -tr nnUNetTrainer_MIND -c 3d_fullres -f 0'
# RFA trainer
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-AMOS_nnUNetTrainer_ParametrizedRFA-0.7 -d 801 -tr nnUNetTrainer_ParametrizedRFA-0.7 -c 3d_fullres -f 0'
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-BCV_nnUNetTrainer_ParametrizedRFA-0.7 -d 801 -tr nnUNetTrainer_ParametrizedRFA-0.7 -c 3d_fullres -f 0'

# RFA trainer (strength=0.0 at prediction)
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.0 && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-AMOS_nnUNetTrainer_ParametrizedRFA-0.0 -d 801 -tr nnUNetTrainer_ParametrizedRFA-0.7 -c 3d_fullres -f 0 --disable_tta'
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.0 && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-BCV_nnUNetTrainer_ParametrizedRFA-0.0 -d 801 -tr nnUNetTrainer_ParametrizedRFA-0.7 -c 3d_fullres -f 0 --disable_tta'


# GIN trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-AMOS_nnUNetTrainer_GIN -d 801 -tr nnUNetTrainer_GIN -c 3d_fullres -f 0'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-BCV_nnUNetTrainer_GIN -d 801 -tr nnUNetTrainer_GIN -c 3d_fullres -f 0'

# 802
# nnUNetTrainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-BCV_nnUNetTrainer -d 802 -tr nnUNetTrainer -c 3d_fullres -f 0'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-AMOS_nnUNetTrainer -d 802 -tr nnUNetTrainer -c3d_fullres -f 0'
# MIND trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-BCV_nnUNetTrainer_MIND -d 802 -tr nnUNetTrainer_MIND -c 3d_fullres -f 0'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-AMOS_nnUNetTrainer_MIND -d 802 -tr nnUNetTrainer_MIND -c3d_fullres -f 0'
# RFA trainer
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-BCV_nnUNetTrainer_ParametrizedRFA-0.7 -d 802 -tr nnUNetTrainer_ParametrizedRFA-0.7 -c 3d_fullres -f 0'
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.7 && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-AMOS_nnUNetTrainer_ParametrizedRFA-0.7 -d 802 -tr nnUNetTrainer_ParametrizedRFA-0.7 -c3d_fullres -f 0'

# RFA trainer (strength=0.0 at prediction)
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.0 && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-BCV_nnUNetTrainer_ParametrizedRFA-0.0 -d 802 -tr nnUNetTrainer_ParametrizedRFA-0.7 -c 3d_fullres -f 0 --disable_tta'
source ./set_envs.sh && NNUNET_RFA_STRENGTH=0.0 && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-AMOS_nnUNetTrainer_ParametrizedRFA-0.0 -d 802 -tr nnUNetTrainer_ParametrizedRFA-0.7 -c3d_fullres -f 0 --disable_tta'

# GIN trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-BCV_nnUNetTrainer_GIN -d 802 -tr nnUNetTrainer_GIN -c 3d_fullres -f 0'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-AMOS_nnUNetTrainer_GIN -d 802 -tr nnUNetTrainer_GIN -c3d_fullres -f 0'

# 803

# nnUNet trainer 2D
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_2d -d 803 -tr nnUNetTrainer -c 2d -f 0 --disable_tta'

# nnUNet trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer -d 803 -tr nnUNetTrainer -c 3d_fullres -f 0 --disable_tta'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-BCV_nnUNetTrainer -d 803 -tr nnUNetTrainer -c 3d_fullres -f 0 --disable_tta'

# insaneDA trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task803_AMOS_w_gallbladder/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task803_AMOS_w_gallbladder-AMOS_nnUNetTrainerV2_insaneDA -t 803 -m 3d_fullres -f 0 -tr nnUNetTrainerV2_insaneDA'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task804_BCV_w_gallbladder/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task803_AMOS_w_gallbladder-BCV_nnUNetTrainerV2_insaneDA -t 803 -m 3d_fullres -f 0 -tr nnUNetTrainerV2_insaneDA'

# insaneDA trainer no mirroring
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task803_AMOS_w_gallbladder/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task803_AMOS_w_gallbladder-AMOS_nnUNetTrainerV2_insaneDAnoMirroring -t 803 -m 3d_fullres -f 0 -tr nnUNetTrainerV2_insaneDAnoMirroring --disable_tta'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task804_BCV_w_gallbladder/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task803_AMOS_w_gallbladder-BCV_nnUNetTrainerV2_insaneDAnoMirroring -t 803 -m 3d_fullres -f 0 -tr nnUNetTrainerV2_insaneDAnoMirroring --disable_tta'

# GIN trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_GIN -d 803 -tr nnUNetTrainer_GIN -c 3d_fullres -f 0 --disable_tta'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-BCV_nnUNetTrainer_GIN -d 803 -tr nnUNetTrainer_GIN -c 3d_fullres -f 0 --disable_tta'


# GIN_MIND trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_GIN_MIND -d 803 -tr nnUNetTrainer_GIN_MIND -c 3d_fullres -f 0 --disable_tta'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-BCV_nnUNetTrainer_GIN_MIND -d 803 -tr nnUNetTrainer_GIN_MIND -c 3d_fullres -f 0 --disable_tta'

# MIND trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_MIND -d 803 -tr nnUNetTrainer_MIND -c 3d_fullres -f 0 --disable_tta'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-BCV_nnUNetTrainer_MIND -d 803 -tr nnUNetTrainer_MIND -c 3d_fullres -f 0 --disable_tta'
# MIND_GIN trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_MIND_GIN -d 803 -tr nnUNetTrainer_MIND_GIN -c 3d_fullres -f 0 --disable_tta'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-BCV_nnUNetTrainer_MIND_GIN -d 803 -tr nnUNetTrainer_MIND_GIN -c 3d_fullres -f 0 --disable_tta'

# nnUNet MIND_MIC

# 1 channel of rnd MIC broadcasted to 12 MIND channels
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_MIND_MIC-np16_dr0.7_dcFalse -d 803 -tr nnUNetTrainer_MIND_MIC-np16_dr0.7_dcFalse -c 3d_fullres -f 0 --mind_mic_dropout 0.7 --mind_mic_num_patches 16 --no-mind_mic_different_per_channel --disable_tta'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-BCV_nnUNetTrainer_MIND_MIC-np16_dr0.7_dcFalse -d 803 -tr nnUNetTrainer_MIND_MIC-np16_dr0.7_dcFalse -c 3d_fullres -f 0 --mind_mic_dropout 0.7 --mind_mic_num_patches 16 --no-mind_mic_different_per_channel --disable_tta'
# 12 channel of rnd MIC for 12 MIND channels
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_MIND_MIC-np16_dr0.7_dcTrue -d 803 -tr nnUNetTrainer_MIND_MIC-np16_dr0.7_dcTrue -c 3d_fullres -f 0 --mind_mic_dropout 0.7 --mind_mic_num_patches 16 --mind_mic_different_per_channel --disable_tta'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-BCV_nnUNetTrainer_MIND_MIC-np16_dr0.7_dcTrue -d 803 -tr nnUNetTrainer_MIND_MIC-np16_dr0.7_dcTrue -c 3d_fullres -f 0 --mind_mic_dropout 0.7 --mind_mic_num_patches 16 --mind_mic_different_per_channel --disable_tta'

# MIC trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_MIC-np16_dr0.7_dcNone -d 803 -tr nnUNetTrainer_MIC-np16_dr0.7_dcNone -c 3d_fullres -f 0 --disable_tta --mic_dropout 0.7 --mic_num_patches 16'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-BCV_nnUNetTrainer_MIC-np16_dr0.7_dcNone -d 803 -tr nnUNetTrainer_MIC-np16_dr0.7_dcNone -c 3d_fullres -f 0 --disable_tta --mic_dropout 0.7 --mic_num_patches 16'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_MIC-np16_dr0.3_dcNone -d 803 -tr nnUNetTrainer_MIC-np16_dr0.3_dcNone -c 3d_fullres -f 0 --disable_tta --mic_dropout 0.3 --mic_num_patches 16'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-BCV_nnUNetTrainer_MIC-np16_dr0.3_dcNone -d 803 -tr nnUNetTrainer_MIC-np16_dr0.3_dcNone -c 3d_fullres -f 0 --disable_tta --mic_dropout 0.3 --mic_num_patches 16'

# 804

# nnUNet trainer 2d
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-BCV_nnUNetTrainer_2d -d 804 -tr nnUNetTrainer -c 2d -f 0 --disable_tta'

# nnUNet trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-BCV_nnUNetTrainer -d 804 -tr nnUNetTrainer -c 3d_fullres -f 0 --disable_tta'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-AMOS_nnUNetTrainer -d 804 -tr nnUNetTrainer -c3d_fullres -f 0 --disable_tta'

# insaneDA trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task804_BCV_w_gallbladder/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task804_BCV_w_gallbladder-BCV_nnUNetTrainerV2_insaneDA -t 804 -m 3d_fullres -f 0 -tr nnUNetTrainerV2_insaneDA'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task803_AMOS_w_gallbladder/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task804_BCV_w_gallbladder-AMOS_nnUNetTrainerV2_insaneDA -t 804 -m 3d_fullres -f 0 -tr nnUNetTrainerV2_insaneDA'

# insaneDA trainer no mirroring
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task804_BCV_w_gallbladder/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task804_BCV_w_gallbladder-BCV_nnUNetTrainerV2_insaneDAnoMirroring -t 804 -m 3d_fullres -f 0 -tr nnUNetTrainerV2_insaneDAnoMirroring --disable_tta'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task803_AMOS_w_gallbladder/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task804_BCV_w_gallbladder-AMOS_nnUNetTrainerV2_insaneDAnoMirroring -t 804 -m 3d_fullres -f 0 -tr nnUNetTrainerV2_insaneDAnoMirroring --disable_tta'

# GIN trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-BCV_nnUNetTrainer_GIN -d 804 -tr nnUNetTrainer_GIN -c 3d_fullres -f 0 --disable_tta'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-AMOS_nnUNetTrainer_GIN -d 804 -tr nnUNetTrainer_GIN -c3d_fullres -f 0 --disable_tta'

# GIN_MIND trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-BCV_nnUNetTrainer_GIN_MIND -d 804 -tr nnUNetTrainer_GIN_MIND -c 3d_fullres -f 0 --disable_tta'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-AMOS_nnUNetTrainer_GIN_MIND -d 804 -tr nnUNetTrainer_GIN_MIND -c3d_fullres -f 0 --disable_tta'
# MIND trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-BCV_nnUNetTrainer_MIND -d 804 -tr nnUNetTrainer_MIND -c 3d_fullres -f 0 --disable_tta'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-AMOS_nnUNetTrainer_MIND -d 804 -tr nnUNetTrainer_MIND -c3d_fullres -f 0 --disable_tta'

# MIND_GIN trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-BCV_nnUNetTrainer_MIND_GIN -d 804 -tr nnUNetTrainer_MIND_GIN -c 3d_fullres -f 0 --disable_tta'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/imagesTs -o /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-AMOS_nnUNetTrainer_MIND_GIN -d 804 -tr nnUNetTrainer_MIND_GIN -c3d_fullres -f 0 --disable_tta'

# MIC trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-BCV_nnUNetTrainer_MIC-np16_dr0.7_dcNone -d 804 -tr nnUNetTrainer_MIC-np16_dr0.7_dcNone -c 3d_fullres -f 0 --disable_tta --mic_dropout 0.7 --mic_num_patches 16'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-AMOS_nnUNetTrainer_MIC-np16_dr0.7_dcNone -d 804 -tr nnUNetTrainer_MIC-np16_dr0.7_dcNone -c3d_fullres -f 0 --disable_tta --mic_dropout 0.7 --mic_num_patches 16'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-BCV_nnUNetTrainer_MIC-np16_dr0.3_dcNone -d 804 -tr nnUNetTrainer_MIC-np16_dr0.3_dcNone -c 3d_fullres -f 0 --disable_tta --mic_dropout 0.3 --mic_num_patches 16'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_predict -i /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/imagesTs -o /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-AMOS_nnUNetTrainer_MIC-np16_dr0.3_dcNone -d 804 -tr nnUNetTrainer_MIC-np16_dr0.3_dcNone -c3d_fullres -f 0 --disable_tta --mic_dropout 0.3 --mic_num_patches 16'

# Evaluation


# 202

source ./set_envs.sh && SYM_PERMUTE_RANGE=full run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task202_AbdomenCTCT/permuted_labelsTs_out -pred /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task202_AbdomenCTCT_permuted/nnUNetTrainerV2_SymPermute-full/connected_compoents -l 1 2 3 4 5 6 7 8 9 10 11 12 13'

source ./set_envs.sh && SYM_PERMUTE_RANGE=six-neighbourhood-only run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task202_AbdomenCTCT/permuted_labelsTs_out -pred /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task202_AbdomenCTCT_permuted/nnUNetTrainerV2_SymPermute-six-neighbourhood-only/connected_compoents -l 1 2 3 4 5 6 7 8 9 10 11 12 13'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task202_AbdomenCTCT/permuted_labelsTs_out -pred /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task202_AbdomenCTCT_permuted/nnUNetTrainerV2_lraspp3d/connected_compoents -l 1 2 3 4 5 6 7 8 9 10 11 12 13'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task202_AbdomenCTCT/permuted_labelsTs_in -pred /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task202_AbdomenCTCT_normal/nnUNetTrainerV2_lraspp3d/connected_compoents -l 1 2 3 4 5 6 7 8 9 10 11 12 13'




source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task650_MMWHS_MRI/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task650_MMWHS_MRI/nnUNetTrainerV2 -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task650_MMWHS_MRI/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task650_MMWHS_MRI/nnUNetTrainerV2_RandomFieldAugmentation -l 1 2 3 4 5 6 7'

# 651
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task651_MMWHS_MRI_HLA/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task651_MMWHS_MRI_HLA/nnUNetTrainerV2 -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task651_MMWHS_MRI_HLA/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task651_MMWHS_MRI_HLA/nnUNetTrainerV2_RandomFieldAugmentation -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task651_MMWHS_MRI_HLA/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task651_MMWHS_MRI_HLA/nnUNetTrainerV2_ParametrizedRandomFieldAugmentation02 -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task651_MMWHS_MRI_HLA/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task651_MMWHS_MRI_HLA/nnUNetTrainerV2_insaneDA -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task651_MMWHS_MRI_HLA/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task651_MMWHS_MRI_HLA/nnUNetTrainerV2_DeepSTAPLE -l 1 2 3 4 5 6 7'


source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task652_MMWHS_MRI_SA/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task652_MMWHS_MRI_SA/nnUNetTrainerV2 -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task652_MMWHS_MRI_SA/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task652_MMWHS_MRI_SA/nnUNetTrainerV2_RandomFieldAugmentation -l 1 2 3 4 5 6 7'


source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task653_ACDC_SA/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task653_ACDC_SA/nnUNetTrainerV2 -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task653_ACDC_SA/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task653_ACDC_SA/nnUNetTrainerV2_RandomFieldAugmentation -l 1 2 3 4 5 6 7'

# 654
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task654_MMWHS_MRI_HLA_PACKED/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task654_MMWHS_MRI_HLA_PACKED/nnUNetTrainerV2_RandomFieldAugmentation -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task654_MMWHS_MRI_HLA_PACKED/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task654_MMWHS_MRI_HLA_PACKED/nnUNetTrainerV2_RandomFieldAugmentation -l 1 2 3 4 5 6 7'

# 655
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task655_MMWHS_COMMON_SPACE/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task655_MMWHS_COMMON_SPACE/nnUNetTrainerV2 -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task655_MMWHS_COMMON_SPACE/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task655_MMWHS_COMMON_SPACE/nnUNetTrainerV2_RandomFieldAugmentation -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task655_MMWHS_COMMON_SPACE/labelsTs -pred /data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task655_MMWHS_COMMON_SPACE/nnUNetTrainerV2_XEdgeConvMax -l 1 2 3 4 5 6 7'


# 656
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task656_MMWHS_RESAMPLE_ONLY/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task656_MMWHS_RESAMPLE_ONLY/nnUNetTrainerV2 -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task656_MMWHS_RESAMPLE_ONLY/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task656_MMWHS_RESAMPLE_ONLY/nnUNetTrainerV2_RandomFieldAugmentation -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task656_MMWHS_RESAMPLE_ONLY/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task656_MMWHS_RESAMPLE_ONLY/nnUNetTrainerV2_XEdgeConvMax -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task656_MMWHS_RESAMPLE_ONLY/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task656_MMWHS_RESAMPLE_ONLY/nnUNetTrainerV2_RandomFieldAugmentationXEdgeConvMax -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task656_MMWHS_RESAMPLE_ONLY/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task656_MMWHS_RESAMPLE_ONLY/nnUNetTrainerV2_RandomFieldAugmentation -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task656_MMWHS_RESAMPLE_ONLY/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task656_MMWHS_RESAMPLE_ONLY_concomp/nnUNetTrainerV2_XEdgeConvMax -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task656_MMWHS_RESAMPLE_ONLY/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task656_MMWHS_RESAMPLE_ONLY/nnUNetTrainerV2_SymPermute-full -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task656_MMWHS_RESAMPLE_ONLY/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task656_MMWHS_RESAMPLE_ONLY/nnUNetTrainerV2_SymPermute-six-neighbourhood-only -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task656_MMWHS_RESAMPLE_ONLY/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task656_MMWHS_RESAMPLE_ONLY/nnUNetTrainerV2_LRASPP3D -l 1 2 3 4 5 6 7'

# 658
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task658_MMWHS_REGISTERED_LOWRES/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task658_MMWHS_REGISTERED_LOWRES/nnUNetTrainerV2_ParametrizedRandomFieldAugmentation -l 1 2 3 4 5'

# 659
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task659_MMWHS_SLICES_HIRES/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task659_MMWHS_SLICES_HIRES/nnUNetTrainerV2_ParametrizedRandomFieldAugmentation -l 1 2 3 4 5'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task659_MMWHS_SLICES_HIRES/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task659_MMWHS_SLICES_HIRES/nnUNetTrainerV2 -l 1 2 3 4 5'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task659_MMWHS_SLICES_HIRES/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task659_MMWHS_SLICES_HIRES_MIUA/nnUNetTrainerV2 -l 1 2 3 4 5'

# 700
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task700_CMRxMotionTask2/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task700_CMRxMotionTask2/nnUNetTrainerV2_RandomFieldAugmentation -l 1 2 3 4 5 6 7'

source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task700_CMRxMotionTask2/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task700_CMRxMotionTask2/nnUNetTrainerV2 -l 1 2 3 4 5 6 7'

# 801

# nnUNetTrainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-AMOS_nnUNetTrainer -l 1 2 3 4 5 6 7 8 9 10 11
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-BCV_nnUNetTrainer -l 1 2 3 4 5 6 7 8 9 10 11

# MIND trainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-AMOS_nnUNetTrainer_MIND -l 1 2 3 4 5 6 7 8 9 10 11
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-BCV_nnUNetTrainer_MIND -l 1 2 3 4 5 6 7 8 9 10 11

# RFA trainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-AMOS_nnUNetTrainer_ParametrizedRFA-0.7 -l 1 2 3 4 5 6 7 8 9 10 11
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-BCV_nnUNetTrainer_ParametrizedRFA-0.7 -l 1 2 3 4 5 6 7 8 9 10 11
# RFA trainer (strength=0.0 at prediction)
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-AMOS_nnUNetTrainer_ParametrizedRFA-0.0 -l 1 2 3 4 5 6 7 8 9 10 11
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-BCV_nnUNetTrainer_ParametrizedRFA-0.0 -l 1 2 3 4 5 6 7 8 9 10 11

# GIN trainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-AMOS_nnUNetTrainer_GIN -l 1 2 3 4 5 6 7 8 9 10 11
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset801_AMOS-BCV_nnUNetTrainer_GIN -l 1 2 3 4 5 6 7 8 9 10 11

# 802

# nnUNetTrainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-AMOS_nnUNetTrainer -l 1 2 3 4 5 6 7 8 9 10 11
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-BCV_nnUNetTrainer -l 1 2 3 4 5 6 7 8 9 10 11

# MIND trainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-AMOS_nnUNetTrainer_MIND -l 1 2 3 4 5 6 7 8 9 10 11
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-BCV_nnUNetTrainer_MIND -l 1 2 3 4 5 6 7 8 9 10 11

# RFA trainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-AMOS_nnUNetTrainer_ParametrizedRFA-0.7 -l 1 2 3 4 5 6 7 8 9 10 11
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-BCV_nnUNetTrainer_ParametrizedRFA-0.7 -l 1 2 3 4 5 6 7 8 9 10 11

# RFA trainer (strength=0.0 at prediction)
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-AMOS_nnUNetTrainer_ParametrizedRFA-0.0 -l 1 2 3 4 5 6 7 8 9 10 11
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-BCV_nnUNetTrainer_ParametrizedRFA-0.0 -l 1 2 3 4 5 6 7 8 9 10 11
# GIN trainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset801_AMOS/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-AMOS_nnUNetTrainer_GIN -l 1 2 3 4 5 6 7 8 9 10 11
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset802_BCV/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset802_BCV-BCV_nnUNetTrainer_GIN -l 1 2 3 4 5 6 7 8 9 10 11


# 803 (evaluation without adrenal glands)

# nnUNet trainer 2d
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_2d -l 1 2 3 4 5 6 7 8 9 10

# nnUNet trainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer -l 1 2 3 4 5 6 7 8 9 10
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-BCV_nnUNetTrainer -l 1 2 3 4 5 6 7 8 9 10

# insaneDA trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task803_AMOS_w_gallbladder/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task803_AMOS_w_gallbladder-AMOS_nnUNetTrainerV2_insaneDA -l 1 2 3 4 5 6 7 8 9 10'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task804_BCV_w_gallbladder/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task803_AMOS_w_gallbladder-BCV_nnUNetTrainerV2_insaneDA -l 1 2 3 4 5 6 7 8 9 10'

# insaneDA trainer no mirroring
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_evaluate_simple /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task803_AMOS_w_gallbladder/labelsTs /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task803_AMOS_w_gallbladder-AMOS_nnUNetTrainerV2_insaneDAnoMirroring -l 1 2 3 4 5 6 7 8 9 10'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_evaluate_simple /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task804_BCV_w_gallbladder/labelsTs /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task803_AMOS_w_gallbladder-BCV_nnUNetTrainerV2_insaneDAnoMirroring -l 1 2 3 4 5 6 7 8 9 10'


# GIN trainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_GIN -l 1 2 3 4 5 6 7 8 9 10
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-BCV_nnUNetTrainer_GIN -l 1 2 3 4 5 6 7 8 9 10

# GIN_MIND trainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_GIN_MIND -l 1 2 3 4 5 6 7 8 9 10
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-BCV_nnUNetTrainer_GIN_MIND -l 1 2 3 4 5 6 7 8 9 10

# MIND trainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_MIND -l 1 2 3 4 5 6 7 8 9 10
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-BCV_nnUNetTrainer_MIND -l 1 2 3 4 5 6 7 8 9 10

# MIND_GIN trainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_MIND_GIN -l 1 2 3 4 5 6 7 8 9 10
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-BCV_nnUNetTrainer_MIND_GIN -l 1 2 3 4 5 6 7 8 9 10

# MIND_MIC trainer
# 1 channel of rnd MIC broadcasted to 12 MIND channels
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_MIND_MIC-np16_dr0.7_dcFalse -l 1 2 3 4 5 6 7 8 9 10
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-BCV_nnUNetTrainer_MIND_MIC-np16_dr0.7_dcFalse -l 1 2 3 4 5 6 7 8 9 10

# 12 channel of rnd MIC for 12 MIND channels
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_MIND_MIC-np16_dr0.7_dcTrue -l 1 2 3 4 5 6 7 8 9 10
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-BCV_nnUNetTrainer_MIND_MIC-np16_dr0.7_dcTrue -l 1 2 3 4 5 6 7 8 9 10

# MIC trainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_MIC-np16_dr0.7_dcNone -l 1 2 3 4 5 6 7 8 9 10
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-BCV_nnUNetTrainer_MIC-np16_dr0.7_dcNone -l 1 2 3 4 5 6 7 8 9 10


source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-AMOS_nnUNetTrainer_MIC-np16_dr0.3_dcNone -l 1 2 3 4 5 6 7 8 9 10
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset803_AMOS_w_gallbladder-BCV_nnUNetTrainer_MIC-np16_dr0.3_dcNone -l 1 2 3 4 5 6 7 8 9 10

# 804 (evaluation without adrenal glands)

# nnUNet trainer 2d
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-BCV_nnUNetTrainer_2d -l 1 2 3 4 5 6 7 8 9 10

# nnUNet trainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-AMOS_nnUNetTrainer -l 1 2 3 4 5 6 7 8 9 10
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-BCV_nnUNetTrainer -l 1 2 3 4 5 6 7 8 9 10

# insaneDA trainer
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task803_AMOS_w_gallbladder/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task804_BCV_w_gallbladder-AMOS_nnUNetTrainerV2_insaneDA -l 1 2 3 4 5 6 7 8 9 10'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNet_evaluate_folder -ref /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task804_BCV_w_gallbladder/labelsTs -pred /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task804_BCV_w_gallbladder-BCV_nnUNetTrainerV2_insaneDA -l 1 2 3 4 5 6 7 8 9 10'

# insaneDA trainer no mirroring
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_evaluate_simple /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task803_AMOS_w_gallbladder/labelsTs /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task804_BCV_w_gallbladder-AMOS_nnUNetTrainerV2_insaneDAnoMirroring -l 1 2 3 4 5 6 7 8 9 10'
source ./set_envs.sh && run_on_recommended_cuda --command 'nnUNetv2_evaluate_simple /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task804_BCV_w_gallbladder/labelsTs /share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task804_BCV_w_gallbladder-BCV_nnUNetTrainerV2_insaneDAnoMirroring -l 1 2 3 4 5 6 7 8 9 10'

# GIN trainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-AMOS_nnUNetTrainer_GIN -l 1 2 3 4 5 6 7 8 9 10
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-BCV_nnUNetTrainer_GIN -l 1 2 3 4 5 6 7 8 9 10

# GIN_MIND trainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-AMOS_nnUNetTrainer_GIN_MIND -l 1 2 3 4 5 6 7 8 9 10
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-BCV_nnUNetTrainer_GIN_MIND -l 1 2 3 4 5 6 7 8 9 10

# MIND trainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-AMOS_nnUNetTrainer_MIND -l 1 2 3 4 5 6 7 8 9 10
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-BCV_nnUNetTrainer_MIND -l 1 2 3 4 5 6 7 8 9 10

# MIND_GIN trainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-AMOS_nnUNetTrainer_MIND_GIN -l 1 2 3 4 5 6 7 8 9 10
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-BCV_nnUNetTrainer_MIND_GIN -l 1 2 3 4 5 6 7 8 9 10

# MIC trainer
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-AMOS_nnUNetTrainer_MIC-np16_dr0.7_dcNone  -l 1 2 3 4 5 6 7 8 9 10
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-BCV_nnUNetTrainer_MIC-np16_dr0.7_dcNone  -l 1 2 3 4 5 6 7 8 9 10


source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset803_AMOS_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-AMOS_nnUNetTrainer_MIC-np16_dr0.3_dcNone  -l 1 2 3 4 5 6 7 8 9 10
source ./set_envs.sh && nnUNetv2_evaluate_simple /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_raw/Dataset804_BCV_w_gallbladder/labelsTs /data_rechenknecht01_2/weihsbach/nnunet/nnUNetV2_results/prediction/Dataset804_BCV_w_gallbladder-BCV_nnUNetTrainer_MIC-np16_dr0.3_dcNone  -l 1 2 3 4 5 6 7 8 9 10