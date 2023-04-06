import os
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from medpy import metric


def hd95(gt, pred):
    return metric.hd95(result=pred, reference=gt, voxelspacing=1.5)


def dice_simi_coeff(pred, gt, logger=None):
    """
    same as the function below

    :param pred:
    :param gt:
    :param logger:
    :return:
    """
    organ_labels = {0: "background", 1: "liver", 2: "spleen", 3: "r_kidney", 4: "l_kidney"}
    dsc = []
    for i in np.unique(pred)[1:]:
        pred_i = np.where(pred != i, 0., pred)
        dsc.append(metric.dc(result=pred_i, reference=gt))
        if logger:
            try:
                logger.info(f"{organ_labels[i]}: {dsc[i - 1] :.3f}")
            except:
                pass
    return dsc


def dice_coeff(outputs, labels, logger=None):
    """
    Evaluation function for Dice score of segmentation overlap
    """
    organ_labels = {0: "background", 1: "liver", 2: "spleen", 3: "r_kidney", 4: "l_kidney"}
    label_nums = np.unique(labels)
    # print("labels:", label_nums)
    dice = []
    for label in label_nums[1:]:
        iflat = (outputs == label).reshape(-1)
        tflat = (labels == label).reshape(-1)
        intersection = (iflat * tflat).sum()
        dsc = (2. * intersection) / (iflat.sum() + tflat.sum())
        if logger:
            try:
                logger.info(f"{organ_labels[label]}: {dsc :.3f}")
            except:
                pass
        dice.append(dsc)
    return np.asarray(dice)
    # return metric.dc(result=outputs, reference=labels)


def Get_Jac(displacement):
    '''
    compute the Jacobian determinant to find out the smoothness of the u.
    refer: https://blog.csdn.net/weixin_41699811/article/details/87691884

    Param: displacement of shape(batch, H, W, D, channel)
    '''
    D_y = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_x = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :])

    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 1])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])

    D = D1 - D2 + D3

    return D


def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """
    if len(disp.shape) not in [3, 4]:
        raise ValueError(f"shape of 2D folw field needs to be [H, W, C]，"
                         f"shape of 3D folw field needs to be [H, W, D, C], but got {disp.shape}")
    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = [np.arange(e) for e in volshape]
    grid_lst = np.meshgrid(*grid_lst, indexing='ij')
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


if __name__ == '__main__':
    from datasets import CT2MRDataset
    from torch.utils.data import DataLoader

    ct2mr_dataset = CT2MRDataset(
        CT_folder=r"F:\shb_src\from_github\datasets\MICCAI2021\task1_preprocessed_by_shb\auxiliary\L2R_Task1_CT",
        CT_name=r"?_bcv_CT.nii.gz",
        MR_folder=r"F:\shb_src\from_github\datasets\MICCAI2021\task1_preprocessed_by_shb\auxiliary\L2R_Task1_MR",
        MR_name=r"?_chaos_MR.nii.gz",
        pair_numbers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
        for_inf=True
    )

    my_dataloader = DataLoader(dataset=ct2mr_dataset, batch_size=1, num_workers=2)
    dice_all_val = np.zeros((len(ct2mr_dataset), 5 - 1))
    for idx, (CTimgs, CTsegs, MRimgs, MRsegs, CTaffines, MRaffines) in enumerate(my_dataloader):
        dice_all_val[idx] = dice_coeff(MRsegs, CTsegs)
    all_val_dice_avgs = dice_all_val.mean(axis=0)
    mean_all_dice = all_val_dice_avgs.mean()
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print(f"dice_all_val: {dice_all_val}, \n "
          f"all_val_dice_avgs: {all_val_dice_avgs}, \n "
          f"mean_all_dice: {mean_all_dice}")

    """
    dice_all_val: 
    [[ 0.768  0.322  0.563  0.592]
     [ 0.452  0.005  0.000  0.000]
     [ 0.039  0.086  0.028  0.001]
     [ 0.315  0.000  0.000  0.000]
     [ 0.470  0.150  0.458  0.317]
     [ 0.678  0.035  0.356  0.000]
     [ 0.501  0.221  0.075  0.179]
     [ 0.505  0.041  0.000  0.000]
     [ 0.705  0.288  0.555  0.109]
     [ 0.643  0.532  0.547  0.495]
     [ 0.421  0.207  0.000  0.000]
     [ 0.484  0.060  0.000  0.002]
     [ 0.513  0.000  0.000  0.000]
     [ 0.660  0.290  0.413  0.249]
     [ 0.680  0.288  0.152  0.075]
     [ 0.612  0.377  0.224  0.084]
     [ 0.610  0.467  0.594  0.353]
     [ 0.569  0.355  0.085  0.143]
     [ 0.739  0.385  0.440  0.242]
     [ 0.482  0.338  0.000  0.023]
     [ 0.652  0.297  0.075  0.073]
     [ 0.727  0.478  0.398  0.180]
     [ 0.592  0.259  0.348  0.000]
     [ 0.322  0.076  0.049  0.023]
     [ 0.431  0.097  0.000  0.000]
     [ 0.382  0.000  0.000  0.000]
     [ 0.271  0.000  0.000  0.000]
     [ 0.717  0.269  0.289  0.153]
     [ 0.674  0.498  0.345  0.316]
     [ 0.541  0.031  0.000  0.048]
     [ 0.398  0.270  0.024  0.226]
     [ 0.478  0.590  0.160  0.297]], 
     all_val_dice_avgs: [ 0.532  0.228  0.193  0.131], 
     mean_all_dice: 0.2711055646981322
    """