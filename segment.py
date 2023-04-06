import numpy as np
import torch
import torch.nn.functional as F
import time
import os
import pathlib
import sys
import nibabel as nib
import scipy.io

import argparse

cuda_idx = 0

from utils.utils import countParam, get_logger
from utils.metrics import dice_coeff
from utils.datasets import SegDataset
from torch.utils.data import DataLoader
from models.obelisk import Seg_Obelisk_Unet
from monai.networks.nets import SwinUNETR


def split_at(s, c, n):
    words = s.split(c)
    return c.join(words[:n]), c.join(words[n:])


def main():
    """
    python inference_seg_lpba.py -image_dir preprocess/datasets/process_cts/pancreas_ct1.nii.gz -output mylabel_ct1.nii.gz -label_dir preprocess/datasets/process_labels/label_ct1.nii.gz
    """
    # read/parse user command line input
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset", choices=["tcia", "chaos", "bcv", "lpba"], default='tcia')

    # choosing backbone network
    parser.add_argument("-backbone", choices=['unet', 'swinunet'], type=str, default='unet',
                        help="if choose swinUNet, image size should be divisible by 2, 4, 8, 32, "
                             "so it's good with size [192, 160, 192], not [144, ...]")
    parser.add_argument("-use_checkpoint",
                        help="use gradient checkpointing for reduced memory usage.", action="store_true")
    parser.add_argument("-feature_size", help="12x feature_size", type=int, default=24)
    parser.add_argument("-use_cuda", help="using cuda or not?",
                        type=lambda s: False if s == "False" else True, default=True)

    parser.add_argument("-model", help="filename of pytorch pth model",
                        default='output/unet_no_ti/tcia_best_0.823.pth', )
    parser.add_argument("-old_model", action="store_true", help="weather I want to load an old model")
    parser.add_argument("-img_transform", choices=['max-min', 'mean-std', 'old-way', 'nope', None],
                        default='old-way',  # max-min for CT/MR
                        type=lambda s: None if s in ['None', 'none', 'nope'] else s,
                        help="what scale type to transform the image")

    parser.add_argument("-image_dir", help="nii.gz CT volume to segment, or dir path for dataloader to load images",
                        default=r"F:\shb_src\from_github\OBELISK\preprocess\datasets\process_cts", )
    parser.add_argument("-img_name",
                        help="prototype scan filename i.e. pancreas_ct?.nii.gz",
                        default="pancreas_ct?.nii.gz")
    parser.add_argument("-label_name", help="prototype segmentation name i.e. label_ct?.nii.gz",
                        default="label_ct?.nii.gz")
    parser.add_argument("-output", help="nii.gz label output prediction",
                        default="output/seg_preds/unet_no_ti/")
    parser.add_argument("-label_dir", help="nii.gz label_dir segmentation, or dir path for dataloader to load segs",
                        default=r"F:\shb_src\from_github\OBELISK\preprocess\datasets\process_labels")
    parser.add_argument("-inf_numbers", help="list of numbers of images for inference",
                        type=lambda s: [str(n) for n in s.split()],
                        default="1 7 17 21 28 33 40 43")
    parser.add_argument("-save_pref", help="prefix of file name to save", type=str, default="")

    args = parser.parse_args()
    d_options = vars(args)

    if not os.path.exists(d_options['output']):
        # os.makedirs(out_dir, exist_ok=True)
        pathlib.Path(d_options['output']).mkdir(parents=True, exist_ok=True)
    logger = get_logger(output=d_options['output'], name=f"{d_options['dataset']}_inference")
    logger.info(f"output to {d_options['output']}")

    ckpt = torch.load(d_options['model'], map_location=torch.device('cpu'))

    if d_options['dataset'] == 'tcia':
        num_labels = 9
        full_res = [144, 144, 144]
    elif d_options['dataset'] in ['bcv', 'chaos']:
        num_labels = 5
        full_res = [192, 160, 192]
    elif d_options['dataset'] == 'lpba':
        num_labels = 55
        full_res = [160, 192, 160]

    # load pretrained OBELISK model
    if args.backbone == 'unet':
        net = Seg_Obelisk_Unet(num_labels, full_res)
    elif args.backbone == 'swinunet':
        net = SwinUNETR(img_size=full_res,
                        in_channels=1,
                        out_channels=num_labels,
                        feature_size=args.feature_size,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=args.use_checkpoint)
    try:
        net.load_state_dict(ckpt["checkpoint"])
    except:
        net.load_state_dict(ckpt["state_dict"])
    # if d_options['old_model']:
    #     net.load_state_dict(obelisk)
    # else:
    #     net.load_state_dict(obelisk["checkpoint"])
    logger.info(f"Successfully loaded model {d_options['model']} with {countParam(net)} parameters")

    net.eval()

    def inference(img_val, seg_val, seg_affine=None, save_name=''):
        if torch.cuda.is_available() == 1 and d_options['use_cuda']:
            print('using GPU acceleration')
            img_val = img_val.cuda()
            net.cuda()
        with torch.no_grad():
            # print(f"image_dir imageval shape: {img_val.shape}")  # torch.Size([1, 1, 144, 144, 144])
            predict = net(img_val)
            # print(f"output predict shape: {predict.shape}")  # torch.Size([1, 9, 144, 144, 144])
            # if d_options['dataset'] == 'visceral':
            #     predict = F.interpolate(predict, size=[D_in0, H_in0, W_in0], mode='trilinear', align_corners=False)

        argmax = torch.argmax(predict, dim=1)
        # print(f"argmax shape: {argmax.shape}")  # torch.Size([1, 144, 144, 144])
        seg_pred = argmax.cpu().short().squeeze().numpy()
        # pred segs: [0 1 2 3 4 5 6 7 8] segs shape: (144, 144, 144)
        seg_img = nib.Nifti1Image(seg_pred, seg_affine)

        save_path = os.path.join(d_options['output'], f"{args.save_pref}pred?_{d_options['dataset']}.nii.gz")
        nib.save(seg_img, save_path.replace("?", save_name))

        if seg_val is not None:
            dice = dice_coeff(torch.from_numpy(seg_pred), seg_val, logger)
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            logger.info(f"Dice validation: {dice}, Avg: {dice.mean() :.3f}")
            # Dice validation: [ 0.939  0.648  0.877  0.808  0.690  0.959  0.914  0.554] Avg. 0.798
        logger.info(f"seged scan number {save_name} save to {d_options['output']} \n")

    if os.path.isfile(d_options['image_dir']):
        img_val = torch.from_numpy(nib.load(d_options['image_dir']).get_fdata()).float().unsqueeze(0).unsqueeze(0)
        img_val = (img_val - img_val.mean()) / img_val.std()  # mean-std scale
        if d_options['label_dir'] is not None:
            seg_val = torch.from_numpy(nib.load(d_options['label_dir']).get_data()).long().unsqueeze(0)
        else:
            seg_val = None
        inference(img_val, seg_val, save_name='')
    elif os.path.isdir(d_options['image_dir']):
        scale_type = "old-way" if d_options['old_model'] else d_options['img_transform']
        test_dataset = SegDataset(image_folder=d_options['image_dir'],
                                  image_name=d_options['img_name'],
                                  label_folder=d_options['label_dir'],
                                  label_name=d_options['label_name'],
                                  scannumbers=d_options['inf_numbers'],
                                  img_transform=scale_type,
                                  for_inf=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)

        for idx, (moving_img, moving_label, img_affine, seg_affine) in enumerate(test_loader):
            inference(moving_img,
                      moving_label,
                      seg_affine=seg_affine.squeeze(0),
                      save_name=str(d_options['inf_numbers'][idx]))


if __name__ == '__main__':
    main()
