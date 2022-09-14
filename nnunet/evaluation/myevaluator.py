from evaluator import nnunet_evaluate_folder
nnunet_evaluate_folder()
    # parser = argparse.ArgumentParser("Evaluates the segmentations located in the folder pred. Output of this script is "
    #                                  "a json file. At the very bottom of the json file is going to be a 'mean' "
    #                                  "entry with averages metrics across all cases")
    # parser.add_argument('-ref', required=True, type=str, help="Folder containing the reference segmentations in nifti "
    #                                                           "format.")
    # parser.add_argument('-pred', required=True, type=str, help="Folder containing the predicted segmentations in nifti "
    #                                                            "format. File names must match between the folders!")
    # parser.add_argument('-l', nargs='+', type=int, required=True, help="List of label IDs (integer values) that should "
    #                                                                    "be evaluated. Best practice is to use all int "
    #                                                                    "values present in the dataset, so for example "
    #                                                                    "for LiTS the labels are 0: background, 1: "
    #                                                                    "liver, 2: tumor. So this argument "
    #                                                                    "should be -l 1 2. You can if you want also "
    #                                                                    "evaluate the background label (0) but in "
    #                                                                    "this case that would not give any useful "
    #                                                                    "information.")