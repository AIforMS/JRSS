import os
import time
import argparse
import pathlib
import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from models.obelisk import Reg_Obelisk_Unet
from monai.networks.nets import SwinUNETR
from models import SpatialTransformer

from utils.utils import get_logger, countParam, LinearWarmupCosineAnnealingLR, ImgTransform, setup_seed
from utils.datasets import RegDataset
# losses
from utils.losses import MIND_loss, NCCLoss, gradient_loss, MutualInformation as MI, TI_Loss
from monai.losses import DiceCELoss, BendingEnergyLoss
from utils.metrics import Get_Jac, dice_coeff
from utils.augment_3d import augment_affine

from utils.visual import visualdl_scalar_lines, visualdl_images, visdom_scalar_lines, visdom_images

setup_seed()


def main():
    # read/parse user command line input
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("-dataset", choices=["bcv", "tcia", "chaos", "ct-mr", "mr-ct"], default='ct-mr')

    parser.add_argument("-mov_folder", help="training moving dataset folder",
                        default=r'F:\shb_src\from_github\datasets\MICCAI2021\task1_preprocessed_by_shb\auxiliary\L2R_Task1_CT\CTs')
    # labels folder is obtained after splicing 'Labels' at the end of the path.
    parser.add_argument("-mov_name", help="prototype scan filename i.e. pancreas_ct?.nii.gz",
                        default='?_bcv_CT.nii.gz')
    parser.add_argument("-fix_folder", help="training fixed dataset folder",
                        default=r'F:\shb_src\from_github\datasets\MICCAI2021\task1_preprocessed_by_shb\auxiliary\L2R_Task1_MR\MRIs')
    parser.add_argument("-fix_name", help="prototype scan filename i.e. pancreas_ct?.nii.gz",
                        default='?_chaos_MR.nii.gz')

    parser.add_argument("-train_pair_numbers",
                        help="list of integers indicating which scans to use, i.e. \"1 2 3\" ",
                        default="11 12 13 14",
                        # 4 7 8 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 34 35 36 37 39
                        type=lambda s: [int(n) for n in s.split()])
    parser.add_argument("-val_pair_numbers",
                        help="list of integers indicating which scans to use, i.e. \"1 2 3\" ",
                        default="1 2",  # 4 5 6 8 9 10
                        type=lambda s: [int(n) for n in s.split()])

    parser.add_argument("-output", help="filename (without extension) for output",
                        default="output/test/")

    # training args
    parser.add_argument("-backbone", choices=['unet', 'swinunet'], type=str, default='swinunet',
                        help="if choose swinUNet, image size should be divisible by 2, 4, 8, 32, "
                             "so it's good with size [192, 160, 192], not [144, ...]")
    parser.add_argument("-use_checkpoint",
                        help="use gradient checkpointing for reduced memory usage when using swinunet.",
                        action="store_true")
    parser.add_argument("-feature_size", help="12x feature_size", type=int, default=24)
    parser.add_argument("-batch_size", help="Dataloader batch size", type=int, default=1)
    parser.add_argument("-lr", help="Optimizer learning rate, keep pace with batch_size",
                        type=float, default=1e-4)  # 0.005 for AdamW, 4e-4 for Adam
    parser.add_argument('-reg_weight', default=1e-5, type=float, help='regularization weight')
    parser.add_argument("-apply_lr_scheduler", help="Need lr scheduler or not", action="store_true")
    parser.add_argument('-warmup_epochs', default=5, type=int, help='number of warmup epochs')
    parser.add_argument("-epochs", help="Train epochs", type=int, default=500)

    parser.add_argument("-pretrained_enc", help="Path to pretrained encoder to initial encoder weight",
                        default=None)  # r"ckpts/model_swinvit_.pth"
    parser.add_argument("-resume", help="Path to pretrained model to continute training",
                        default=None)  # r"ckpts/swin_unetr.base_5000ep_f48_lr2e-4_pretrained_.pth"
    parser.add_argument("-from_ckpt", help="Get star_epoch and best_acc from ckpt?", action="store_true")

    parser.add_argument("-interval", help="validation and ckpt saving interval", type=int, default=1)
    parser.add_argument("-is_visdom", help="Using Visdom to visualize Training process",
                        type=lambda s: False if s == "False" else True, default=False)
    parser.add_argument("-is_visualdl", help="Using visualDL to visualize Training process",
                        type=lambda s: False if s == "False" else True, default=False)
    parser.add_argument("-num_workers", help="Dataloader num_workers", type=int, default=2)

    # losses args
    parser.add_argument("-on_ROI", help="Apply sim loss on ROI based od seg labels",
                        action="store_true")
    parser.add_argument("-weakly_sup", help="if apply weakly supervised, use reg dice loss, else not",
                        action="store_true")
    parser.add_argument("-apply_ti_loss", help="if apply TI loss, Topological Interactions for Multi-Class",
                        action="store_true")
    parser.add_argument("-seg_loss", help="choose loss func for seg label supervised criterion", choices=['dce', 'ce'],
                        default="ce")
    parser.add_argument("-sim_loss", type=str, help="similarity criterion",
                        choices=['MIND', 'MSE', 'NCC', 'MI'], default='MIND')
    # loss weight
    parser.add_argument("-alpha", type=float, help="weight for regularization loss",
                        default=0.025)  # ncc: 1.5, mse: 0.025, MIND-SSC: 4.0, VM: 0.01
    parser.add_argument("-seg_weight", help="Seg loss weight", type=float, default=0.1)
    parser.add_argument("-sim_weight", help="Similarity loss weight", type=float, default=1.0)
    parser.add_argument("-ti_weight", help="TI loss weight", type=float, default=1e-5)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_epoch, steps, best_acc = 0, 0, 0

    if not os.path.exists(args.output):
        pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

    logger = get_logger(args.output)

    if args.is_visdom:
        logger.info("visdom starting, need to open the server: python -m visdom.server")
    if args.is_visualdl:
        logger.info(f"visualDL starting, need to open the server: visualdl --logdir {args.output}")

    if args.weakly_sup:
        logger.info(f"Weakly supervised training with {args.seg_loss} loss and {args.sim_loss} loss")
    if args.on_ROI:
        logger.info("Using ROI based on seg labels for sim loss")

    logger.info(f"Training on dataset {args.dataset} with backbone {args.backbone}, output to {args.output}")

    # load train images and segmentations
    logger.info(f'train pair_numbers: {args.train_pair_numbers}')

    train_dataset = RegDataset(mov_folder=args.mov_folder,
                               mov_name=args.mov_name,
                               fix_folder=args.fix_folder,
                               fix_name=args.fix_name,
                               pair_numbers=args.train_pair_numbers,
                               img_transform=ImgTransform("max-min"))

    val_dataset = RegDataset(mov_folder=args.mov_folder,
                             mov_name=args.mov_name,
                             fix_folder=args.fix_folder,
                             fix_name=args.fix_name,
                             pair_numbers=args.val_pair_numbers,
                             img_transform=ImgTransform("max-min"))

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=args.num_workers)

    end_epoch = args.epochs  # 300

    num_labels = train_dataset.get_labels_num()
    logger.info(f"num of labels: {num_labels}")

    H = train_dataset.H
    W = train_dataset.W
    D = train_dataset.D

    class_weight = train_dataset.get_class_weight()
    class_weight = class_weight / class_weight.mean()
    class_weight[0] = 0.15
    class_weight = class_weight.cuda()
    logger.info(f'inv sqrt class_weight {class_weight}')

    # initialise trainable network parts
    if args.backbone == 'unet':
        reg_net = Reg_Obelisk_Unet((H, W, D))
        # optimizer and lr_scheduler initial after network initial
        optimizer = optim.Adam(reg_net.parameters(), lr=args.lr, weight_decay=1e-5)
    elif args.backbone == 'swinunet':
        reg_net = SwinUNETR(img_size=(H, W, D),
                            in_channels=2,
                            out_channels=3,
                            feature_size=args.feature_size,
                            drop_rate=0.0,
                            attn_drop_rate=0.0,
                            dropout_path_rate=0.0,
                            use_checkpoint=args.use_checkpoint)
        optimizer = optim.AdamW(reg_net.parameters(), lr=args.lr, weight_decay=1e-5)

    reg_net.to(device)
    reg_net.train()

    stn_val = SpatialTransformer(size=[H, W, D])
    stn_val.to(device)

    logger.info(f"Reg Net total params: {countParam(reg_net)}")

    if args.resume:
        model_dict = torch.load(args.resume)
        reg_net.load_state_dict(model_dict["state_dict"], strict=False)
        if args.from_ckpt:
            start_epoch, steps, best_acc = model_dict['epoch'], model_dict['steps'], model_dict['best_acc']
        logger.info(f"Training resume from {args.resume}, start from ckpt? {args.from_ckpt}")

    if args.pretrained_enc:
        try:
            weight = torch.load(args.pretrained_enc)
            reg_net.load_from(weights=weight)
            logger.info('Using pretrained self-supervied Swin UNETR backbone weights !')
        except ValueError:
            raise ValueError('Self-supervised pre-trained weights not available')

    # train using Adam with weight decay and exponential LR decay
    scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                              warmup_epochs=args.warmup_epochs,
                                              max_epochs=args.epochs) if args.apply_lr_scheduler else None
    if args.resume:
        try:
            model_dict = torch.load(args.resume)
            optimizer.load_state_dict(model_dict['optimizer'])
            if args.apply_lr_scheduler and args.from_ckpt:
                scheduler.load_state_dict(model_dict["scheduler"])
                scheduler.step(epoch=start_epoch)
            logger.info(f"Optimizer continue")
        except:
            pass

    # losses
    if args.sim_loss == "MIND":
        sim_criterion = MIND_loss
    elif args.sim_loss == "MSE":
        sim_criterion = nn.MSELoss()
    elif args.sim_loss == "NCC":
        sim_criterion = NCCLoss()
    elif args.sim_loss == "MI":
        sim_criterion = MI()

    if args.apply_ti_loss:
        labels = np.arange(num_labels)[1:]  # exclude bg 0
        exclusion = list(itertools.combinations(labels, 2))  # [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
        ti_criterion = TI_Loss(dim=3, connectivity=6, inclusion=[], exclusion=exclusion)
    else:
        ti_criterion = None

    grad_criterion = gradient_loss  # BendingEnergyLoss()、gradient_loss

    if args.weakly_sup:
        if args.seg_loss == 'ce':
            seg_criterion = nn.CrossEntropyLoss(class_weight)
        else:
            seg_criterion = DiceCELoss(to_onehot_y=True, softmax=True, ce_weight=class_weight)
    else:
        seg_criterion = None

    run_loss = np.zeros([end_epoch, 5])
    dsc_list = np.zeros((len(val_dataset), num_labels - 1))
    dice_all_val_bi = np.zeros((len(val_dataset), num_labels - 1))
    batch_nums = len(train_loader)
    logger.info(f'Training set sizes: {len(train_dataset)}, Train loader size: {batch_nums}, '
                f'Validation set sizes: {len(val_dataset)}')

    # run for 1000 iterations / 250 epochs
    for epoch in range(start_epoch, end_epoch):

        t0 = time.time()

        # select random training pair (mini-batch=4 averaging at the end)
        for mov_imgs, mov_segs, fix_imgs, fix_segs in train_loader:
            steps += 1

            if np.random.choice([0, 1]):
                # 50% to apply data augment
                with torch.no_grad():
                    mov_imgs, mov_segs = augment_affine(mov_imgs.to(device), mov_segs.to(device), strength=0.0375)
                    fix_imgs, fix_segs = augment_affine(fix_imgs.to(device), fix_segs.to(device), strength=0.0375)
                    torch.cuda.empty_cache()
            else:
                mov_imgs, mov_segs = mov_imgs.to(device), mov_segs.to(device)
                fix_imgs, fix_segs = fix_imgs.to(device), fix_segs.to(device)

            # run forward registration: x -> y
            ddf = reg_net(torch.cat([mov_imgs, fix_imgs], dim=1))

            # grid_sample 用最近邻插值的话梯度会是 0。用线性插值的话，不能直接插原 label，要先 one-hot。
            mov_segs_one_hot = F.one_hot(
                mov_segs.squeeze(1), num_classes=num_labels).permute(0, 4, 1, 2, 3).float()  # NxNum_LabelsxHxWxD

            moved_segs = stn_val(mov_segs_one_hot, ddf)  # 采用线性插值对seg进行warped
            moved_imgs = stn_val(mov_imgs, ddf)

            sim_loss = sim_criterion(moved_imgs, fix_imgs)
            seg_loss = seg_criterion(moved_segs, fix_segs.squeeze(1)) if seg_criterion else torch.tensor(0)
            ti_loss = ti_criterion(mov_segs_one_hot, fix_segs) if ti_criterion else torch.tensor(0)

            grad_loss = grad_criterion(ddf)

            total_loss = args.sim_weight * sim_loss \
                         + args.alpha * grad_loss \
                         + args.seg_weight * seg_loss \
                         + args.ti_weight * ti_loss

            run_loss[epoch, 0] += total_loss.item()
            run_loss[epoch, 1] += args.sim_weight * sim_loss.item()
            run_loss[epoch, 2] += args.alpha * grad_loss.item()
            run_loss[epoch, 3] += args.seg_weight * seg_loss.item()
            run_loss[epoch, 4] += args.ti_weight * ti_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            del total_loss
            torch.cuda.empty_cache()

        scheduler.step() if scheduler else None  # epoch wise lr scheduler

        time_t = time.time() - t0

        # verbose ON: report some numbers and run inference on (potentially unseen test images)
        if epoch % args.interval == 0:
            reg_net.eval()
            Jac_std, Jac_neg = [], []

            for val_idx, (mov_img, mov_seg, fix_img, fix_seg) in enumerate(val_loader):
                mov_img, mov_seg, fix_img, fix_seg = mov_img.to(device), mov_seg.to(device), fix_img.to(device), fix_seg.to(device)
                t0 = time.time()
                with torch.no_grad():

                    # if segmentations are available for some validation/training data, Dice can be computed
                    ddf = reg_net(torch.cat([mov_img, fix_img], dim=1))
                    moved_seg = stn_val(mov_seg.float(), ddf, mode="nearest")

                    time_i = time.time() - t0

                    dsc_list[val_idx] = dice_coeff(fix_seg.cpu(), moved_seg.long().cpu())

                    torch.cuda.empty_cache()

                    # complexity of transformation and foldings
                    jacdet = Get_Jac(ddf.permute(0, 2, 3, 4, 1)).cpu()
                    Jac_std.append(jacdet.std())
                    Jac_neg.append(100 * ((jacdet <= 0.).sum() / jacdet.numel()))

                    if args.is_visualdl:
                        visualdl_images(args, step=val_idx, seg_moved=moved_seg, seg_fixed=fix_seg)

            # logger some feedback information
            dsc_avg_list = dsc_list.mean(axis=0)
            dsc_avg_all = dsc_avg_list.mean()
            latest_lr = optimizer.state_dict()['param_groups'][0]['lr']

            is_best = dsc_avg_all > best_acc
            best_acc = max(dsc_avg_all, best_acc)

            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            run_loss /= batch_nums
            logger.info(
                f"Validating: epoch {epoch}, step {steps}, time train {round(time_t, 3)}, time infer {round(time_i, 3)}, "
                f"total loss {run_loss[epoch, 0] :.3f}, sim loss {run_loss[epoch, 1] :.3f}, "
                f"grad loss {run_loss[epoch, 2] :.3f}, seg loss {run_loss[epoch, 3] :.3f}, "
                f"ti loss {run_loss[epoch, 4] :.3f}, stdJac {np.mean(Jac_std) :.3f}, Jac<=0 {np.mean(Jac_neg) :.3f}%, "
                f"dsc_avg_list {dsc_avg_list}, dsc avg {dsc_avg_all :.3f}, best_acc {best_acc :.3f}, "
                f"lr {latest_lr :.8f}")

            if args.is_visdom:
                visdom_scalar_lines(args, epoch, losses=run_loss, accs=dsc_avg_list, lr=latest_lr)

            if args.is_visualdl:
                visualdl_scalar_lines(args, epoch, losses=run_loss, accs=dsc_avg_list,
                                      best_acc=best_acc, lr=latest_lr, accs_bi=None)

            reg_net.cpu()

            state_dict = {
                "state_dict": reg_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if args.apply_lr_scheduler else None,
                "best_acc": best_acc,
                "epoch": epoch,
                "steps": steps
            }

            if is_best:
                np.save(f"{args.output}run_loss.npy", run_loss)
                torch.save(state_dict,
                           args.output + f"{args.dataset}_best_{round(best_acc, 3) if best_acc > 0.65 else ''}.pth")
                logger.info(f"saved the best model at epoch {epoch}, with best acc {best_acc :.3f}")
            else:
                torch.save(state_dict, args.output + f"{args.dataset}_latest.pth")

            reg_net.train()
            reg_net.to(device)


if __name__ == '__main__':
    main()
