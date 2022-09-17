from typing import Tuple, List, Union
from skimage import io, segmentation, morphology, measure, exposure
import SimpleITK as sitk
import numpy as np
import tifffile
from batchgenerators.utilities.file_and_folder_operations import *

def convert_2d_segmentation_nifti_to_img(nifti_file: str, output_filename: str, transform=None, export_dtype=np.uint8):
    img = sitk.GetArrayFromImage(sitk.ReadImage(nifti_file))
    assert img.shape[0] == 1, "This function can only export 2D segmentations!"
    img = img[0]
    if transform is not None:
        img = transform(img)
    img[img==1]=255
    # img = measure.label(img)
    io.imsave(output_filename, img.astype(export_dtype), check_contrast=False)
    # tifffile.imwrite(output_filename, img.astype(np.uint32), compression='zlib')
convert_2d_segmentation_nifti_to_img('/media/oem/sda21/wxg/codebase/dataset/nnUNet_trained_models/nnUNet/2d/Task300_SSLSeg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/cell_00001_label.nii.gz','/media/oem/sda21/wxg/codebase/dataset/nnUNet_trained_models/nnUNet/2d/Task300_SSLSeg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/cell_00001_label.png')

def convert_3d_segmentation_nifti_to_tiff(nifti_file: str, output_filename: str, transform=None, export_dtype=np.uint8):
    img = sitk.GetArrayFromImage(sitk.ReadImage(nifti_file))
    assert len(img.shape) == 3, "This function can only export 3D segmentations!"
    if transform is not None:
        img = transform(img)

    tifffile.imsave(output_filename, img.astype(export_dtype))

# if __name__ == '__main__':
#     base = '/media/oem/sda21/wxg/codebase/dataset/nnUNet_trained_models/nnUNet/2d/Task527_LUNGSeg7/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw'
#     out = '/media/oem/sda21/wxg/codebase/dataset/nnUNet_trained_models/nnUNet/2d/Task527_LUNGSeg7/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/pngs'
#     training_cases = subfiles(base, join=False)
#     for i in training_cases:
#         if 'nii.gz' in i:
#             print(i)
#             nifti_file = join(base,i)
#             output_filename = join(out,i.replace('.nii.gz','.png'))
#             convert_2d_segmentation_nifti_to_img(nifti_file,output_filename)




