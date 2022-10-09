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


import nnunet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
# from nnunet.experiment_planning.utils import crop
from nnunet.paths import *
import shutil
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.preprocessing.sanity_checks import verify_dataset_integrity
from nnunet.training.model_restore import recursive_find_python_class
from nnunet.preprocessing.cropping import ImageCropper
from nnunet.preprocessing.my_croppingV2 import ImageCropper_2
from nnunet.configuration import default_num_threads
from multiprocessing import Pool

import SimpleITK as sitk
import nibabel as nib
import numpy as np

def main():
    '''
    define my preprocess for unlabeled data
    :TODO verify crop code to adapt new data
    :TODO then revise train_SSL class code to solve problems

    '''
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_ids", nargs="+", default= [307],help="List of integers belonging to the task ids you wish to run"
                                                            " experiment planning and preprocessing for. Each of these "
                                                            "ids must, have a matching folder 'TaskXXX_' in the raw "
                                                            "data folder")
    parser.add_argument("-pl3d", "--planner3d", type=str, default="None",
                        help="Name of the ExperimentPlanner class for the full resolution 3D U-Net and U-Net cascade. "
                             "Default is ExperimentPlanner3D_v21. Can be 'None', in which case these U-Nets will not be "
                             "configured")
    parser.add_argument("-pl2d", "--planner2d", type=str, default="ExperimentPlanner2D_v21",
                        help="Name of the ExperimentPlanner class for the 2D U-Net. Default is ExperimentPlanner2D_v21. "
                             "Can be 'None', in which case this U-Net will not be configured")
    parser.add_argument("-no_pp", action="store_true",
                        help="Set this flag if you dont want to run the preprocessing. If this is set then this script "
                             "will only run the experiment planning and create the plans file")
    parser.add_argument("-tl", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the low resolution data for the 3D low "
                             "resolution U-Net. This can be larger than -tf. Don't overdo it or you will run out of "
                             "RAM")
    parser.add_argument("-tf", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the full resolution data of the 2D U-Net and "
                             "3D U-Net. Don't overdo it or you will run out of RAM")
    parser.add_argument("--verify_dataset_integrity", required=False, default=True, action="store_true",
                        help="set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")
    parser.add_argument("-overwrite_plans", type=str, default=None, required=False,
                        help="Use this to specify a plans file that should be used instead of whatever nnU-Net would "
                             "configure automatically. This will overwrite everything: intensity normalization, "
                             "network architecture, target spacing etc. Using this is useful for using pretrained "
                             "model weights as this will guarantee that the network architecture on the target "
                             "dataset is the same as on the source dataset and the weights can therefore be transferred.\n"
                             "Pro tip: If you want to pretrain on Hepaticvessel and apply the result to LiTS then use "
                             "the LiTS plans to run the preprocessing of the HepaticVessel task.\n"
                             "Make sure to only use plans files that were "
                             "generated with the same number of modalities as the target dataset (LiTS -> BCV or "
                             "LiTS -> Task008_HepaticVessel is OK. BraTS -> LiTS is not (BraTS has 4 input modalities, "
                             "LiTS has just one)). Also only do things that make sense. This functionality is beta with"
                             "no support given.\n"
                             "Note that this will first print the old plans (which are going to be overwritten) and "
                             "then the new ones (provided that -no_pp was NOT set).")
    parser.add_argument("-overwrite_plans_identifier", type=str, default=None, required=False,
                        help="If you set overwrite_plans you need to provide a unique identifier so that nnUNet knows "
                             "where to look for the correct plans and data. Assume your identifier is called "
                             "IDENTIFIER, the correct training command would be:\n"
                             "'nnUNet_train CONFIG TRAINER TASKID FOLD -p nnUNetPlans_pretrained_IDENTIFIER "
                             "-pretrained_weights FILENAME'")

    args = parser.parse_args()
    task_ids = args.task_ids
    dont_run_preprocessing = args.no_pp
    tl = args.tl
    tf = args.tf
    planner_name3d = args.planner3d
    planner_name2d = args.planner2d

    if planner_name3d == "None":
        planner_name3d = None
    if planner_name2d == "None":
        planner_name2d = None

    if args.overwrite_plans is not None:
        if planner_name2d is not None:
            print("Overwriting plans only works for the 3d planner. I am setting '--planner2d' to None. This will "
                  "skip 2d planning and preprocessing.")
        assert planner_name3d == 'ExperimentPlanner3D_v21_Pretrained', "When using --overwrite_plans you need to use " \
                                                                       "'-pl3d ExperimentPlanner3D_v21_Pretrained'"
    
    def create_lists_from_splitted_dataset(base_folder_splitted):
        lists = []

        json_file = join(base_folder_splitted, "dataset.json")
        with open(json_file) as jsn:
            d = json.load(jsn)
            training_files = d['training']
        num_modalities = len(d['modality'].keys())
        for tr in training_files:
            cur_pat = []
            for mod in range(num_modalities):
                cur_pat.append(join(base_folder_splitted, "imagesTr", tr['image'].split("/")[-1][:-7] +
                                    "_%04.0d.nii.gz" % mod))
            cur_pat.append(None)    ### no GT
            lists.append(cur_pat)
            ########cur_pat list -> [train_pth,..., label_pth]#################
        return lists, {int(i): d['modality'][str(i)] for i in d['modality'].keys()}
    
    def crop(task_string, override=False, num_threads=default_num_threads):
        cropped_out_dir = join(nnUNet_cropped_data, task_string)
        maybe_mkdir_p(cropped_out_dir)

        if override and isdir(cropped_out_dir):
            shutil.rmtree(cropped_out_dir)
            maybe_mkdir_p(cropped_out_dir)

        splitted_4d_output_dir_task = join(nnUNet_raw_data, task_string)
        lists, _ = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

        imgcrop = ImageCropper_2(num_threads, cropped_out_dir)
        #######ImageCropper是主要的实现crop类，对无标签数据crop需要改写一下#######
        imgcrop.run_cropping(lists, overwrite_existing=override)
        shutil.copy(join(nnUNet_raw_data, task_string, "dataset.json"), cropped_out_dir)
    def verify_all_same_orientation(folder):
        """
        This should run after cropping
        :param folder:
        :return:
        """
        nii_files = subfiles(folder, suffix=".nii.gz", join=True)
        orientations = []
        for n in nii_files:
            img = nib.load(n)
            affine = img.affine
            orientation = nib.aff2axcodes(affine)
            orientations.append(orientation)
        # now we need to check whether they are all the same
        orientations = np.array(orientations)
        unique_orientations = np.unique(orientations, axis=0)
        all_same = len(unique_orientations) == 1
        return all_same, unique_orientations
    def verify_dataset_integrity(folder):
        """
        folder needs the imagesTr, imagesTs and labelsTr subfolders. There also needs to be a dataset.json
        checks if all training cases and labels are present
        checks if all test cases (if any) are present
        for each case, checks whether all modalities apre present
        for each case, checks whether the pixel grids are aligned
        checks whether the labels really only contain values they should
        :param folder:
        :return:
        """
        assert isfile(join(folder, "dataset.json")), "There needs to be a dataset.json file in folder, folder=%s" % folder
        assert isdir(join(folder, "imagesTr")), "There needs to be a imagesTr subfolder in folder, folder=%s" % folder
        assert isdir(join(folder, "labelsTr")), "There needs to be a labelsTr subfolder in folder, folder=%s" % folder
        dataset = load_json(join(folder, "dataset.json"))
        training_cases = dataset['training']
        num_modalities = len(dataset['modality'].keys())
        test_cases = dataset['test']
        expected_train_identifiers = [i['image'].split("/")[-1][:-7] for i in training_cases]
        expected_test_identifiers = [i.split("/")[-1][:-7] for i in test_cases]

        ## check training set
        nii_files_in_imagesTr = subfiles((join(folder, "imagesTr")), suffix=".nii.gz", join=False)
        nii_files_in_labelsTr = subfiles((join(folder, "labelsTr")), suffix=".nii.gz", join=False)

        label_files = []
        geometries_OK = True
        has_nan = False

        # check all cases
        if len(expected_train_identifiers) != len(np.unique(expected_train_identifiers)): raise RuntimeError("found duplicate training cases in dataset.json")

        print("Verifying training set")
        for c in expected_train_identifiers:
            print("checking case", c)
            # check if all files are present
            expected_label_file = join(folder, "labelsTr", c + ".nii.gz")
            label_files.append(expected_label_file)
            expected_image_files = [join(folder, "imagesTr", c + "_%04.0d.nii.gz" % i) for i in range(num_modalities)]
            # assert isfile(expected_label_file), "could not find label file for case %s. Expected file: \n%s" % (
            #     c, expected_label_file)
            assert all([isfile(i) for i in
                        expected_image_files]), "some image files are missing for case %s. Expected files:\n %s" % (
                c, expected_image_files)

            # verify that all modalities and the label have the same shape and geometry.
            # label_itk = sitk.ReadImage(expected_label_file)

            # nans_in_seg = np.any(np.isnan(sitk.GetArrayFromImage(label_itk)))
            # has_nan = has_nan | nans_in_seg
            # if nans_in_seg:
            #     print("There are NAN values in segmentation %s" % expected_label_file)

            images_itk = [sitk.ReadImage(i) for i in expected_image_files]
            for i, img in enumerate(images_itk):
                nans_in_image = np.any(np.isnan(sitk.GetArrayFromImage(img)))
                has_nan = has_nan | nans_in_image
                # same_geometry = verify_same_geometry(img, label_itk)
                # if not same_geometry:
                #     geometries_OK = False
                #     print("The geometry of the image %s does not match the geometry of the label file. The pixel arrays "
                #         "will not be aligned and nnU-Net cannot use this data. Please make sure your image modalities "
                #         "are coregistered and have the same geometry as the label" % expected_image_files[0][:-12])
                if nans_in_image:
                    print("There are NAN values in image %s" % expected_image_files[i])

            # now remove checked files from the lists nii_files_in_imagesTr and nii_files_in_labelsTr
            for i in expected_image_files:
                nii_files_in_imagesTr.remove(os.path.basename(i))
            # nii_files_in_labelsTr.remove(os.path.basename(expected_label_file))

        # check for stragglers
        assert len(
            nii_files_in_imagesTr) == 0, "there are training cases in imagesTr that are not listed in dataset.json: %s" % nii_files_in_imagesTr
        # assert len(
        #     nii_files_in_labelsTr) == 0, "there are training cases in labelsTr that are not listed in dataset.json: %s" % nii_files_in_labelsTr

        # verify that only properly declared values are present in the labels
        print("Verifying label values---SKIP")
######对于无标签数据跳过数据检查这一步#######
        if len(expected_test_identifiers) > 0:
            print("Verifying test set")
            nii_files_in_imagesTs = subfiles((join(folder, "imagesTs")), suffix=".nii.gz", join=False)

            for c in expected_test_identifiers:
                # check if all files are present
                expected_image_files = [join(folder, "imagesTs", c + "_%04.0d.nii.gz" % i) for i in range(num_modalities)]
                assert all([isfile(i) for i in
                            expected_image_files]), "some image files are missing for case %s. Expected files:\n %s" % (
                    c, expected_image_files)

                # verify that all modalities and the label have the same geometry. We use the affine for this
                if num_modalities > 1:
                    images_itk = [sitk.ReadImage(i) for i in expected_image_files]
                    reference_img = images_itk[0]

                    # for i, img in enumerate(images_itk[1:]):
                    #     assert verify_same_geometry(img, reference_img), "The modalities of the image %s do not seem to be " \
                    #                                                     "registered. Please coregister your modalities." % (
                    #                                                         expected_image_files[i])

                # now remove checked files from the lists nii_files_in_imagesTr and nii_files_in_labelsTr
                for i in expected_image_files:
                    nii_files_in_imagesTs.remove(os.path.basename(i))
            assert len(
                nii_files_in_imagesTs) == 0, "there are training cases in imagesTs that are not listed in dataset.json: %s" % nii_files_in_imagesTr

        all_same, unique_orientations = verify_all_same_orientation(join(folder, "imagesTr"))
        if not all_same:
            print(
                "WARNING: Not all images in the dataset have the same axis ordering. We very strongly recommend you correct that by reorienting the data. fslreorient2std should do the trick")
        # save unique orientations to dataset.json
        if not geometries_OK:
            raise Warning("GEOMETRY MISMATCH FOUND! CHECK THE TEXT OUTPUT! This does not cause an error at this point  but you should definitely check whether your geometries are alright!")
        else:
            print("TRAIN no label Dataset OK")

        if has_nan:
            raise RuntimeError("Some images have nan values in them. This will break the training. See text output above to see which ones")

    # we need raw data
    tasks = []
    for i in task_ids:
        i = int(i)

        task_name = convert_id_to_task_name(i)
#########跳过这一步完整性检查，下一步就是crop数据，path:raw_data->raw_data/cropped_data#######
        if args.verify_dataset_integrity:
            verify_dataset_integrity(join(nnUNet_raw_data, task_name))

        crop(task_name, False, tf)

        tasks.append(task_name)

    search_in = join(nnunet.__path__[0], "experiment_planning")

    if planner_name3d is not None:
        planner_3d = recursive_find_python_class([search_in], planner_name3d, current_module="nnunet.experiment_planning")
        if planner_3d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "nnunet.experiment_planning" % planner_name3d)
    else:
        planner_3d = None

    if planner_name2d is not None:
        planner_2d = recursive_find_python_class([search_in], planner_name2d, current_module="nnunet.experiment_planning")
        if planner_2d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "nnunet.experiment_planning" % planner_name2d)
    else:
        planner_2d = None

    for t in tasks:
        print("\n\n\n", t)
        cropped_out_dir = os.path.join(nnUNet_cropped_data, t)
        preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t)
        #splitted_4d_output_dir_task = os.path.join(nnUNet_raw_data, t)
        #lists, modalities = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

        # we need to figure out if we need the intensity propoerties. We collect them only if one of the modalities is CT
        dataset_json = load_json(join(cropped_out_dir, 'dataset.json'))
        modalities = list(dataset_json["modality"].values())
        collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False
        dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=tf)  # this class creates the fingerprint
        _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner


        maybe_mkdir_p(preprocessing_output_dir_this_task)
        shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
        shutil.copy(join(nnUNet_raw_data, t, "dataset.json"), preprocessing_output_dir_this_task)

        threads = (tl, tf)

        print("number of threads: ", threads, "\n")

        if planner_3d is not None:
            if args.overwrite_plans is not None:
                assert args.overwrite_plans_identifier is not None, "You need to specify -overwrite_plans_identifier"
                exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task, args.overwrite_plans,
                                         args.overwrite_plans_identifier)
            else:
                exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task)
            exp_planner.plan_experiment()
            if not dont_run_preprocessing:  # double negative, yooo
                exp_planner.run_preprocessing(threads)
        if planner_2d is not None:
            exp_planner = planner_2d(cropped_out_dir, preprocessing_output_dir_this_task)
            exp_planner.plan_experiment()
            if not dont_run_preprocessing:  # double negative, yooo
                exp_planner.run_preprocessing(threads)

if __name__ == "__main__":
    main()

