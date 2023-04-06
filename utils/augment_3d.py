import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
import math


def random_rotate3D(img_numpy, min_angle, max_angle):
    """
    Returns a random rotated array in the same shape
    :param img_numpy: 3D numpy array
    :param min_angle: in degrees
    :param max_angle: in degrees
    :return: 3D rotated img
    """
    assert img_numpy.ndim == 3, "provide a 3d numpy array"
    assert min_angle < max_angle, "min should be less than max val"
    assert min_angle > -360 or max_angle < 360
    all_axes = [(1, 0), (1, 2), (0, 2)]
    angle = np.random.randint(low=min_angle, high=max_angle + 1)
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]
    return ndimage.rotate(img_numpy, angle, axes=axes)


class RandomRotation:
    def __init__(self, min_angle=-10, max_angle=10):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, img_numpy, label=None):
        """
        Args:
            img_numpy (numpy): Image to be rotated.
            label (numpy): Label segmentation map to be rotated

        Returns:
            img_numpy (numpy): rotated img.
            label (numpy): rotated Label segmentation map.
        """
        img_numpy = random_rotate3D(img_numpy, self.min_angle, self.max_angle)
        if label.any() != None:
            label = random_rotate3D(label, self.min_angle, self.max_angle)
        return img_numpy, label


def augment_affine(img_in, seg_in, mind_in=None, strength=2.5):
    """
    3D affine augmentation on image and segmentation mini-batch on GPU.
    (affine transf. is centered: trilinear interpolation and zero-padding used for sampling)
    :param input: img_in batch (torch.cuda.FloatTensor), seg_in batch (torch.cuda.LongTensor)
    :return: augmented BxCxTxHxW image batch (torch.cuda.FloatTensor), augmented BxTxHxW seg batch (torch.cuda.LongTensor)
    """
    # use_what = np.random.choice([0])  # 1, 2,
    B, C, D, H, W = img_in.size()

    # if use_what == 0:
    # 仿射变形
    affine_matrix = (torch.eye(3, 4).unsqueeze(0) + torch.randn(B, 3, 4) * strength).to(img_in.device)

    # elif use_what == 1:
    #     # 缩放
    #     z = np.random.choice([0.8, 0.9, 1.1, 1.2])
    #     affine_matrix = torch.tensor([[z, 0, 0, 0],
    #                                   [0, z, 0, 0],
    #                                   [0, 0, z, 0]], dtype=torch.float32).to(img_in.device)
    #
    # elif use_what == 2:
    #     # 旋转
    #     angle = np.random.choice([-10, -5, 5, 10]) * math.pi / 180
    #     affine_matrix = torch.tensor([[math.cos(angle), math.sin(-angle), math.sin(-angle), 0],
    #                                   [math.sin(angle), math.cos(angle), math.sin(-angle), 0],
    #                                   [math.sin(angle), math.sin(angle), math.cos(angle), 0]],
    #                                  dtype=torch.float32).to(img_in.device)

    meshgrid = F.affine_grid(affine_matrix.expand(B, 3, 4), size=[B, 1, D, H, W], align_corners=False)

    img_out = F.grid_sample(img_in, meshgrid, padding_mode='border', align_corners=False)
    seg_out = F.grid_sample(seg_in.float(), meshgrid, mode='nearest', align_corners=False).long()
    if mind_in:
        mind_out = F.grid_sample(mind_in, meshgrid, padding_mode='border', align_corners=False)
        return img_out, seg_out, mind_out  # .type(dtype=torch.uint8) 保存成 nii 需要 uint8类型
    else:
        return img_out, seg_out


if __name__ == "__main__":
    import nibabel as nib
    from torch.utils.data import DataLoader
    from preprocess.which_scale_to_apply import MyDataset, LPBADataset

    my_dataset = MyDataset(image_folder="../preprocess/datasets/process_cts/",
                           image_name="pancreas_ct?.nii.gz",
                           label_folder="../preprocess/datasets/process_labels",
                           label_name="label_ct?.nii.gz",
                           scannumbers=[43])
    # my_dataset = LPBADataset(
    #     image_folder=r"D:\code_sources\from_github\Medical Images Seg & Reg\MICCAI2020\vm_troch\dataset\LPBA40\train",
    #     image_name="S?.delineation.skullstripped.nii.gz",
    #     label_folder=r"D:\code_sources\from_github\Medical Images Seg & Reg\MICCAI2020\vm_troch\dataset\LPBA40\label",
    #     label_name="S?.delineation.structure.label.nii.gz",
    #     scannumbers=[12]
    # )
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1)
    print(f"len dataset: {len(my_dataset)}, len dataloader: {len(my_dataloader)}")
    imgs, segs, img_affine, seg_affine = next(iter(my_dataloader))
    print(f"imgs size: {imgs.size()}, segs size: {segs.size()}")
    print(f"min, max in imgs: {torch.min(imgs), torch.max(imgs)}")
    print(f"mean, std in imgs: {imgs.mean(), imgs.std()}")

    imgs, segs = augmentAffine(img_in=imgs, seg_in=segs, strength=0.075)

    img_np = imgs.squeeze().squeeze().numpy()
    seg_np = segs.squeeze().numpy()
    # sitk.WriteImage(sitk.GetImageFromArray(img_np), "img_lpba.nii.gz")
    # sitk.WriteImage(sitk.GetImageFromArray(seg_np.astype(np.uint8)), "seg_lpba.nii.gz")
    img_nib = nib.Nifti1Image(img_np, affine=img_affine.squeeze().numpy())
    seg_nib = nib.Nifti1Image(seg_np, affine=seg_affine.squeeze().numpy())
    nib.save(img_nib, 'img_tcia.nii.gz')
    nib.save(seg_nib, 'seg_tcia.nii.gz')
