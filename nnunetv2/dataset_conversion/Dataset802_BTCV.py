import glob
import shutil

from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def convert_btcv(nnunet_dataset_id: int = 802):
    data_dir = join('/mnt/share/nnunet/nnUNet_raw/Dataset802_BTCV/')

    training_images = glob.glob(os.path.join(data_dir, 'imagesTr/*.nii.gz'))
    for case in training_images:
        shutil.move(case, os.path.join(data_dir, 'imagesTr/' + 'case_' + case[-11:-7] + '_0000.nii.gz'))

    training_labels = glob.glob(os.path.join(data_dir, 'labelsTr/*.nii.gz'))
    for case in training_labels:
        shutil.move(case, os.path.join(data_dir, 'labelsTr/' + 'case_' + case[-11:]))

    generate_dataset_json(data_dir,
                          {0: "CT"},
                          labels={
                              "background": 0,
                              'spleen': 1,
                              'right_kidney': 2,
                              'left_kidney': 3,
                              'gallblader': 4,
                              'esophagus': 5,
                              'liver': 6,
                              'stomach': 7,
                              'aorta': 8,
                              'inferior_vena_cava': 9,
                              'portal_and_splenic_veins': 10,
                              'pancreas': 11,
                              'right_adrenal_gland': 12,
                              'left_adrenal_gland': 13
                          },
                          num_training_cases=30,
                          file_ending='.nii.gz',
                          dataset_name='BTCV'
                          )


if __name__ == '__main__':
    convert_btcv()


