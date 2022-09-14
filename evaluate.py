import os
import numpy as np
import SimpleITK as sitk
import tqdm
import torch.utils.data
from glob import glob
from hausdorff import hausdorff_distance
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"


def dice_coef(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    #output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    #target = target.view(-1).data.cpu().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

infer_path = "/media/oem/sda21/wxg/codebase/valdata/12/output/"  # 推理结果地址
label_path = "/media/oem/sda21/wxg/codebase/valdata/12/labelsTs/"  # 测试集label地址

# 因为是脑肿瘤数据，所以这里将三个标签0，1，2转为wt、tc、et三个区域
def wt_tc_et_make(npmask):
    WT_Label = npmask.copy()
    WT_Label[npmask == 1] = 1
    WT_Label[npmask == 2] = 1
    WT_Label[npmask == 3] = 1
    TC_Label = npmask.copy()
    TC_Label[npmask == 1] = 1
    TC_Label[npmask == 2] = 0
    TC_Label[npmask == 3] = 1
    ET_Label = npmask.copy()
    ET_Label[npmask == 1] = 0
    ET_Label[npmask == 2] = 0
    ET_Label[npmask == 3] = 1
    # nplabel = np.empty((240, 240, 3))#之前切成160 现在临时改成240
    # nplabel = np.empty((160, 160, 3))
    nplabel = np.empty((npmask.shape[0], npmask.shape[1], 3))
    nplabel[:, :, 0] = WT_Label
    nplabel[:, :, 1] = TC_Label
    nplabel[:, :, 2] = ET_Label
    nplabel = nplabel.transpose((2, 0, 1))
    del npmask
    return nplabel

def visit_data():
    wt_dices = []
    tc_dices = []
    et_dices = []
    dices = [wt_dices, tc_dices, et_dices]

    wt_hd = []
    tc_hd = []
    et_hd = []
    hds = [wt_hd, tc_hd, et_hd]
    image_list = os.listdir(label_path)
    for i in tqdm.tqdm(range(len(image_list))):
        pred_nii = sitk.ReadImage(infer_path + image_list[i], sitk.sitkUInt8)
        pred_arr = sitk.GetArrayFromImage(pred_nii)[0, :, :]
        pred_cu = np.array(pred_arr)
        pred = wt_tc_et_make(pred_cu)

        mask_nii = sitk.ReadImage(label_path + image_list[i], sitk.sitkUInt8)
        mask_arr = sitk.GetArrayFromImage(mask_nii)[0, :, :]
        mask_cu = np.array(mask_arr)
        mask = wt_tc_et_make(mask_cu)
        for j in range(3):
            dice = dice_coef(pred[j, :, :], mask[j, :, :])
            hd = hausdorff_distance(pred[j, :, :], mask[j, :, :])
            dices[j].append(dice)
            hds[j].append(hd)
        del pred_nii, pred, pred_cu, pred_arr, mask_nii, mask, mask_cu, mask_arr
    dices = np.array(dices)
    hds = np.array(hds)

    print(f"wt dice is {np.mean(dices[0, :])}")
    print(f"tc dice is {np.mean(dices[1, :])}")
    print(f"et dice is {np.mean(dices[2, :])}")
    print(f"wt hd is {np.mean(hds[0, :])}")
    print(f"tc hd is {np.mean(hds[1, :])}")
    print(f"et hd is {np.mean(hds[2, :])}")

if __name__ == '__main__':
    visit_data()
