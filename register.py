import os
import argparse
import pathlib
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader

from utils.metrics import dice_coeff
from utils.utils import ImgTransform, get_logger
from utils.datasets import MultiModalDataset
from models import VxmDense, SpatialTransformer
from monai.networks.nets import SwinUNETR

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument("-mov_folder", help="training moving dataset folder",
                    default=r'F:\shb_src\from_github\datasets\MICCAI2021\task1_preprocessed_by_shb\auxiliary\L2R_Task1_CT\CTs')
parser.add_argument("-mov_name", help="prototype scan filename i.e. pancreas_ct?.nii.gz",
                    default='img00?_tcia_CT.nii.gz')
parser.add_argument("-fix_folder", help="training fixed dataset folder",
                    default=r'F:\shb_src\from_github\datasets\MICCAI2021\task1_preprocessed_by_shb\auxiliary\L2R_Task1_MR\MRIs')
parser.add_argument("-fix_name", help="prototype scan filename i.e. pancreas_ct?.nii.gz",
                    default='img00?_tcia_MR.nii.gz')
parser.add_argument("-test_pair_numbers",
                    help="list of integers indicating which scans to use, i.e. \"1 2 3\" ",
                    default="02 04 06 08 10 12 14 16",
                    type=lambda s: [str(n) for n in s.split()])

parser.add_argument('-output', default=r"output/reg_results",
                    help="warped results output folder name, NOTE: no '/' at the end!")
parser.add_argument('-gpu', default=1, type=int, help='whether to use gpu')
parser.add_argument('-model_name', default="unet", choices=['unet', 'swin_unetr'],
                    help='choose which model for nonlinear registration')
parser.add_argument('-model', default=r"output/pair_wise_mind_mind/bcv_best_518_.pth",
                    help='The pretrained model parameters for registration infer.')

parser.add_argument("-bidir", help="negate flow for bidirectional reg",
                    type=lambda s: False if s == "False" else True, default=False)

parser.add_argument("-img_size", help="list of image size, i.e. \"192 160 192\" ",
                    default="192 160 192",
                    type=lambda s: [int(n) for n in s.split()])
parser.add_argument("-use_checkpoint", dest="use_checkpoint",
                    help="use gradient checkpointing for reduced memory usage.", action="store_true")
parser.add_argument("-feature_size", dest="feature_size", help="12x feature_size", type=int, default=24)
parser.add_argument("-num_workers", help="Dataloader num_workers", type=int, default=2)

args = parser.parse_args()

if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on GPU")
else:
    device = torch.device('cpu')
    print("Running on CPU")

if not os.path.exists(args.output):
    best_acc = os.path.split(args.model)[-1].split('_')[-1][:-4]
    args.output = args.output + best_acc
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

logger = get_logger(args.output, name='test')

if args.bidir:
    logger.info(f"Loading model from {args.model}, register MR to CT")
else:
    logger.info(f"Loading model from {args.model}, register CT to MR")


def unet_reg(mov_img, fix_img, mov_seg, fix_seg):
    # load and set up model
    reg_net = VxmDense.load(args.model, device)
    reg_net.to(device)
    reg_net.eval()

    # predict
    with torch.no_grad():
        moved_img, moved_seg, ddf = reg_net(mov_img, fix_img, mov_seg=mov_seg, bidir=args.bidir)

    return moved_img, moved_seg, ddf


def swin_unet_reg(mov_img, fix_img, mov_seg, fix_seg):
    # initialise trainable network parts
    reg_net = SwinUNETR(img_size=args.img_size,
                        in_channels=2,
                        out_channels=3,
                        feature_size=args.feature_size,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=args.use_checkpoint)
    model_dict = torch.load(args.model)["state_dict"]
    reg_net.load_state_dict(model_dict)
    reg_net.eval()
    reg_net.to(device)

    stn_val = SpatialTransformer(size=args.img_size)
    stn_val.to(device)

    # predict
    with torch.no_grad():
        ddf = reg_net(torch.cat([mov_img, fix_img], dim=1))
        if args.bidir:
            ddf = -ddf
        else:
            moved_img = stn_val(mov_img, ddf)
            moved_seg = stn_val(mov_seg.float(), ddf, mode="nearest")

    return moved_img, moved_seg, ddf


def main():
    # load images
    numbers = args.test_pair_numbers

    test_dataset = MultiModalDataset(mov_folder=args.mov_folder,
                                     mov_name=args.mov_name,
                                     fix_folder=args.fix_folder,
                                     fix_name=args.fix_name,
                                     pair_numbers=numbers,
                                     img_transform=ImgTransform("max-min"),
                                     for_inf=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=args.num_workers)

    affine = test_dataset.affine

    for idx, (mov_img, mov_seg, fix_img, fix_seg) in enumerate(test_loader):

        mov_img, mov_seg, fix_img, fix_seg = mov_img.to(device), mov_seg.to(device), fix_img.to(device), fix_seg.to(device)

        # initial dsc
        DSC_init = dice_coeff(mov_seg.long().cpu(), fix_seg.long().cpu())

        if args.model_name == 'unet':
            moved_img, moved_seg, ddf = unet_reg(mov_img, fix_img, mov_seg, fix_seg)
        elif args.model_name == 'swin_unetr':
            moved_img, moved_seg, ddf = swin_unet_reg(mov_img, fix_img, mov_seg, fix_seg)

        if args.bidir:
            fix_seg = mov_seg.long().cpu()
        else:
            fix_seg = fix_seg.long().cpu()

        # evaluate
        DSC = dice_coeff(moved_seg.long().cpu(), fix_seg, logger=logger)

        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        logger.info(f"Initial DSC: {DSC_init}, DSC Avg.: {DSC_init[DSC_init != 0.].mean()}")
        logger.info(f"Registered DSC: {DSC}, DSC Avg.: {DSC[DSC != 0.].mean()}")

        # save moved image
        if args.bidir:
            # MR -> CT
            file_name = "MR" + numbers[idx]
        else:
            # CT -> MR
            file_name = "CT" + numbers[idx]

        nib.save(nib.Nifti1Image(moved_img.cpu().squeeze().numpy(), affine),
                 os.path.join(args.output, f"warped_{file_name}.nii.gz"))
        nib.save(nib.Nifti1Image(ddf.cpu().permute(0, 2, 3, 4, 1).squeeze().numpy(), affine),
                 os.path.join(args.output, f"flow_{file_name}.nii.gz"))
        nib.save(nib.Nifti1Image(moved_seg.cpu().short().squeeze().numpy(), affine),
                 os.path.join(args.output, f"warped_seg_{file_name}.nii.gz"))

        if not args.bidir:
            logger.info(f"CT img {numbers[idx]} was warped to MR img {numbers[idx]} \n")
        else:
            logger.info(
                f"Using negate flow for bidirectional reg, MR img {numbers[idx]} was warped to CT img {numbers[idx]} \n")


if __name__ == "__main__":
    main()
