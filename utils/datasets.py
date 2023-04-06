import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
from utils import ImgTransform


class RegDataset(Dataset):
    def __init__(self,
                 mov_folder,
                 mov_name,
                 fix_folder,
                 fix_name,
                 pair_numbers,
                 img_transform=ImgTransform(scale_type="max-min"),
                 for_inf=False):
        super(RegDataset, self).__init__()

        self.for_inf = for_inf

        if mov_name.find("?") == -1 or fix_name.find("?") == -1:
            raise ValueError('error! filename must contain \"?\" to insert your chosen numbers')

        if len(pair_numbers) == 0:
            raise ValueError(f"You have to choose which pair numbers [list] to be train")

        self.mov_imgs, self.mov_segs = [], []
        self.fix_imgs, self.fix_segs = [], []
        for i in pair_numbers:
            if self.for_inf:
                mov_img_name = mov_name.replace("?", str(i))
                mov_label_name = mov_name.replace("?", str(i)).replace("img", "seg")
                mov_img_nib = nib.load(os.path.join(mov_folder, mov_img_name))
                mov_seg_nib = nib.load(os.path.join(mov_folder, mov_label_name))

                fix_img_name = fix_name.replace("?", str(i))
                fix_label_name = fix_name.replace("?", str(i)).replace("img", "seg")
                fix_img_nib = nib.load(os.path.join(fix_folder, fix_img_name))
                fix_seg_nib = nib.load(os.path.join(fix_folder, fix_label_name))
            else:
                # loading CT
                mov_img_name = mov_name.replace("?", f"img{str(i)}")
                mov_label_name = mov_name.replace("?", f"seg{str(i)}")
                mov_img_nib = nib.load(os.path.join(mov_folder, mov_img_name))
                mov_seg_nib = nib.load(os.path.join(mov_folder, mov_label_name))
                # loading MR
                fix_img_name = fix_name.replace("?", f"img{str(i)}")
                fix_label_name = fix_name.replace("?", f"seg{str(i)}")
                fix_img_nib = nib.load(os.path.join(fix_folder, fix_img_name))
                fix_seg_nib = nib.load(os.path.join(fix_folder, fix_label_name))

            self.affine = mov_img_nib.affine
            mov_img = mov_img_nib.get_fdata()
            fix_img = fix_img_nib.get_fdata()

            if img_transform is not None:
                mov_img = img_transform(mov_img)
                fix_img = img_transform(fix_img)

            self.mov_imgs.append(torch.from_numpy(mov_img).unsqueeze(0).unsqueeze(0).float())
            self.mov_segs.append(torch.from_numpy(mov_seg_nib.get_fdata()).unsqueeze(0).unsqueeze(0).long())
            self.fix_imgs.append(torch.from_numpy(fix_img).unsqueeze(0).unsqueeze(0).float())
            self.fix_segs.append(torch.from_numpy(fix_seg_nib.get_fdata()).unsqueeze(0).unsqueeze(0).long())

        self.mov_imgs = torch.cat(self.mov_imgs, 0)
        self.mov_segs = torch.cat(self.mov_segs, 0)
        self.fix_imgs = torch.cat(self.fix_imgs, 0)
        self.fix_segs = torch.cat(self.fix_segs, 0)

        self.len_ = min(len(self.mov_imgs), len(self.fix_imgs))

        img_shape = self.mov_segs[0].shape
        self.H, self.W, self.D = img_shape[-3], img_shape[-2], img_shape[-1]

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        fix_idx = np.random.choice(np.where(np.arange(self.len_) != idx)[0])  # 随机选取一个 fixed image

        if not self.for_inf:
            # random pair-wised multi-modal registration
            return self.mov_imgs[idx], self.mov_segs[idx], self.fix_imgs[fix_idx], self.fix_segs[fix_idx]
        else:
            # 预测时需要mov_name和fix_name一一对应
            return self.mov_imgs[idx], self.mov_segs[idx], self.fix_imgs[idx], self.fix_segs[idx]

    def get_class_weight(self):
        mov_weight = torch.sqrt(1.0 / (torch.bincount(self.mov_segs.view(-1)).float()))
        fix_weight = torch.sqrt(1.0 / (torch.bincount(self.fix_segs.view(-1)).float()))
        # print(f"mov weight: {mov_weight}, fix weight: {fix_weight}")
        return (mov_weight + fix_weight) / 2

    def get_labels_num(self):
        return int(len(self.mov_segs[0].unique()))


class SegDataset(Dataset):
    def __init__(self,
                 image_folder,
                 image_name,
                 label_folder,
                 label_name,
                 scannumbers,
                 img_transform="max-min",
                 for_inf=False):
        super(SegDataset, self).__init__()
        self.for_inf = for_inf

        if image_name.find("?") == -1 or label_name.find("?") == -1:
            raise ValueError('error! filename must contain \"?\" to insert your chosen numbers')

        if len(scannumbers) == 0:
            raise ValueError(f"You have to choose which scannumbers [list] to be train")

        self.imgs, self.segs, self.img_affines, self.seg_affines = [], [], [], []
        for i in scannumbers:
            # /share/data_rechenknecht01_1/heinrich/TCIA_CT
            filescan1 = image_name.replace("?", str(i))
            img_nib = nib.load(os.path.join(image_folder, filescan1))
            self.img_affines.append(img_nib.affine)
            img = img_nib.get_fdata()
            if img_transform is not None:
                img_trans = ImgTransform(scale_type=img_transform)
                # scale img in mean-std way
                img = img_trans(img)

            fileseg1 = label_name.replace("?", str(i))
            seg_nib = nib.load(os.path.join(label_folder, fileseg1))
            self.seg_affines.append(seg_nib.affine)

            self.imgs.append(torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float())
            self.segs.append(torch.from_numpy(seg_nib.get_fdata()).unsqueeze(0).unsqueeze(0).long())

        self.imgs = torch.cat(self.imgs, 0)
        # self.imgs = (self.imgs - self.imgs.mean()) / self.imgs.std()  # mean-std scale
        # self.imgs = (self.imgs - self.imgs.min()) / (self.imgs.max() - self.imgs.min())  # max-min scale to [0, 1]
        # self.imgs = self.imgs / 1024.0 + 1.0  # raw data scale to [0, 3]
        self.segs = torch.cat(self.segs, 0)
        self.img_affines = np.stack(self.img_affines)
        self.seg_affines = np.stack(self.seg_affines)
        self.len_ = min(len(self.imgs), len(self.segs))

        img_shape = self.segs[0].shape
        self.H, self.W, self.D = img_shape[-3], img_shape[-2], img_shape[-1]

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        if not self.for_inf:
            return self.imgs[idx], self.segs[idx]
        else:
            return self.imgs[idx], self.segs[idx], self.img_affines[idx], self.seg_affines[idx]

    def get_class_weight(self):
        return torch.sqrt(1.0 / (torch.bincount(self.segs.view(-1)).float()))

    def get_labels_num(self):
        return int(len(self.segs[0].unique()))


if __name__ == '__main__':
    my_dataset = RegDataset(
        mov_folder=r"F:\shb_src\from_github\datasets\MICCAI2021\task1_preprocessed_by_shb\auxiliary\L2R_Task1_CT",
        mov_name=r"?_bcv_CT.nii.gz",
        fix_folder=r"F:\shb_src\from_github\datasets\MICCAI2021\task1_preprocessed_by_shb\auxiliary\L2R_Task1_MR",
        fix_name=r"?_chaos_MR.nii.gz",
        pair_numbers=[1, 5, 17, 21, 39, 31, 24],
        img_transform=ImgTransform(scale_type="max-min"),
        for_inf=True
    )

    my_dataloader = DataLoader(dataset=my_dataset, batch_size=2, num_workers=2)

    print(f"len dataset: {len(my_dataset)}, len dataloader: {len(my_dataloader)}")
    CTimgs, CTsegs, MRimgs, MRsegs, CTaffines, MRaffines = next(iter(my_dataloader))
    print(f"CT imgs size: {CTimgs.size()}, CT segs size: {CTsegs.size()}, "
          f"MR imgs size: {MRimgs.size()}, MR segs size: {MRsegs.size()}")
    print(f"min, max in CT imgs: {torch.min(CTimgs), torch.max(CTimgs)}, "
          f"min, max in MR imgs: {torch.min(MRimgs), torch.max(MRimgs)}")
    print(f"mean, std in CT imgs: {CTimgs.mean(), CTimgs.std()}, "
          f"mean, std in MR imgs: {MRimgs.mean(), MRimgs.std()}")
    print(f"CT affines == MR affines: {(CTaffines == MRaffines).all()}")  # True
    """
    affine:
    [[2., 0., 0., 0.],
    [0., 2., 0., 0.],
    [0., 0., 2., 0.],
    [0., 0., 0., 1.]]
    """
    # mean, std in CT imgs: (tensor(0.1342), tensor(0.2189)), mean, std in MR imgs: (tensor(0.0480), tensor(0.0963))
    print(f"class weights: {my_dataset.get_class_weight()}")
    # class weights: tensor([0.0002, 0.0008, 0.0019, 0.0024, 0.0024])
    print(f"num of labels: {my_dataset.get_labels_num()}")

    class_weight = my_dataset.get_class_weight()
    class_weight = class_weight / class_weight.mean()
    class_weight[0] = 0.5
    print(f"class weights_: {class_weight}")
    # class weights_: tensor([0.5000, 0.5140, 1.2697, 1.5493, 1.5622])
