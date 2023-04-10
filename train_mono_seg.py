import itertools
import time
import os
import visdom
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.utils import init_weights, countParam, get_cosine_schedule_with_warmup, get_logger, setup_seed
from utils.metrics import dice_coeff
from utils.augment_3d import augment_affine
from utils.datasets import SingleModalDataset
from utils.losses import OHEMLoss, TI_Loss
from models.obelisk import Seg_Obelisk_Unet
from monai.networks.nets import SwinUNETR

setup_seed()


def main():
    # read/parse user command line input
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset", choices=["tcia", "chaos", "bcv", "lpba"], default='tcia')
    parser.add_argument("-imgfolder", help="training images dataset folder",
                        default=r'F:\shb_src\from_github\OBELISK\preprocess\datasets\process_cts')
    parser.add_argument("-labelfolder", help="training labels dataset folder",
                        default=r'F:\shb_src\from_github\OBELISK\preprocess\datasets\process_labels')
    parser.add_argument("-scannumbers",
                        help="list of integers indicating which scans to use, i.e. \"1 2 3\" ",
                        default="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 23 24 25 28 29 "
                                "30 31 32 34 35 36 37 38 39 40 41 42 43",
                        type=lambda s: [str(n) for n in s.split()])
    parser.add_argument("-val_scannumbers",
                        help="list of integers indicating which scans to use, i.e. \"1 2 3\" ",
                        default="1 7 17 21 28 33 40 43",
                        type=lambda s: [str(n) for n in s.split()])
    parser.add_argument("-filescan",
                        help="prototype scan filename i.e. pancreas_ct?.nii.gz",
                        default='pancreas_ct?.nii.gz')  # pancreas_ct?.nii.gz
    parser.add_argument("-fileseg", help="prototype segmentation name i.e. label_ct?.nii.gz",
                        default="label_ct?.nii.gz")
    parser.add_argument("-output", help="filename (without extension) for output",
                        default="output/obelisunet_ti/")
    parser.add_argument("-img_transform", help="image transform way", choices=['mean-std', 'max-min', 'old-way',
                                                                               'nope', None],
                        type=lambda s: None if s == 'nope' else s, default="nope")

    # choosing backbone network
    parser.add_argument("-backbone", choices=['unet', 'swinunet'], type=str, default='unet',
                        help="if choose swinUNet, image size should be divisible by 2, 4, 8, 32, "
                             "so it's good with size [192, 160, 192], not [144, ...]")
    parser.add_argument("-use_checkpoint",
                        help="use gradient checkpointing for reduced memory usage.", action="store_true")
    parser.add_argument("-feature_size", help="12x feature_size", type=int, default=24)

    # training args
    parser.add_argument("-num_workers", help="Dataloader num_workers", type=int, default=2)
    parser.add_argument("-batch_size", help="Dataloader batch size", type=int, default=2)
    parser.add_argument("-lr", help="Optimizer learning rate, keep pace with batch_size",
                        type=float, default=0.001)  # 1e-4 for chaos and bcv, with lr scheduler
    parser.add_argument("-apply_lr_scheduler", help="Need lr scheduler or not", action="store_true")
    parser.add_argument("-warmup_steps", help="step for Warmup scheduler", type=int, default=5)

    # Loss
    parser.add_argument("-apply_ti_loss", help="Need TI Loss or not", action="store_true")
    parser.add_argument("-ti_weight", help="TI loss weight", type=float, default=1e-6)
    parser.add_argument("-ce_weight", help="criterion loss weight", type=float, default=1.0)
    parser.add_argument("-criterion", help="Which loss to train", choices=['ohem', 'ce'], type=str, default='ohem')

    parser.add_argument("-epochs", help="Train epochs", type=int, default=350)  # 2000 for swinUnet
    parser.add_argument("-resume", help="Path to ourself pretrained model to continute training", default=None)
    parser.add_argument("-from_ckpt", help="Get star_epoch and best_acc from ckpt?", action="store_true")
    parser.add_argument("-pretrained", help="Path to official pretrained model to continute training", default=None)
    parser.add_argument("-interval", help="validation and saving interval", type=int, default=1)
    parser.add_argument("-visdom", help="Using Visdom to visualize Training process",
                        type=lambda s: False if s == "False" else True, default=False)

    args = parser.parse_args()
    d_options = vars(args)
    is_visdom = d_options["visdom"]

    best_acc = 0
    star_epoch = 0
    end_epoch = d_options['epochs']  # 300

    if not os.path.exists(d_options['output']):
        os.mkdir(d_options['output'])

    logger = get_logger(d_options['output'])
    # sys.stdout = Logger(d_options['output'] + 'log.txt')
    logger.info(f"Training on dataset {args.dataset} with backbone {d_options['backbone']}, "
                f"and loss {args.criterion}, output to {d_options['output']}")

    if is_visdom:
        vis = visdom.Visdom()  # using visdom
        logger.info("visdom starting, open the server: python -m visdom.server")
        loss_opts = {'xlabel': 'epochs',
                     'ylabel': 'loss',
                     'title': 'Loss Line',
                     'legend': ['total loss', 'ti loss', 'ce loss']}
        acc_opts = {'xlabel': 'epochs',
                    'ylabel': 'acc',
                    'title': 'Acc Line',
                    'legend': ['1 spleen', '2 pancreas', '3 kidney', '4 gallbladder', '5 esophagus', '6 liver',
                               '7 stomach', '8 duodenum'] if d_options['dataset'] == 'tcia'
                    else ['1 liver', '2 spleen', '3 right kidney', '4 left kidney']}
        lr_opts = {'xlabel': 'epochs', 'ylabel': 'lr', 'title': 'LR Line'}
        best_acc_opt = {'xlabel': 'epochs', 'ylabel': 'best acc', 'title': 'Best Acc Line'}

    # load train images and segmentations
    val_scannumbers = d_options['val_scannumbers']
    train_scannumbers = [i for i in d_options['scannumbers'] if i not in val_scannumbers]
    logger.info(f'train scannumbers: {train_scannumbers}')
    if d_options['filescan'].find("?") == -1:
        raise ValueError('error filescan must contain \"?\" to insert numbers')

    file_imgs = d_options['filescan']
    file_labels = d_options['fileseg']
    img_transform = d_options['img_transform']

    train_dataset = SingleModalDataset(image_folder=d_options['imgfolder'],
                                       image_name=file_imgs,
                                       label_folder=d_options['labelfolder'],
                                       label_name=file_labels,
                                       scannumbers=train_scannumbers,
                                       img_transform=img_transform)

    val_dataset = SingleModalDataset(image_folder=d_options['imgfolder'],
                                     image_name=file_imgs,
                                     label_folder=d_options['labelfolder'],
                                     label_name=file_labels,
                                     scannumbers=val_scannumbers,
                                     img_transform=img_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=d_options['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=2)

    class_weight = train_dataset.get_class_weight()
    class_weight = class_weight / class_weight.mean()
    class_weight[0] = 0.5
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    logger.info(f'inv sqrt class_weight: {class_weight.data.cpu().numpy()}')
    # [ 0.50  0.59  1.13  0.73  1.96  2.80  0.24  0.46  1.00]

    num_labels = int(class_weight.numel())
    logger.info(f"num of labels: {num_labels}")

    if args.criterion == 'ohem':
        criterion = OHEMLoss(0.25, class_weight.cuda())  # Online Hard Example Mining Loss ~= Soft CELoss
    elif args.criterion == 'ce':
        criterion = nn.CrossEntropyLoss(class_weight.cuda())

    if d_options['apply_ti_loss']:
        labels = np.arange(num_labels)[1:]
        exclusion = list(itertools.combinations(labels, 2))  # [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
        ti_criterion = TI_Loss(dim=3, connectivity=6, inclusion=[], exclusion=exclusion)
    else:
        ti_criterion = None

    full_res = [train_dataset.H, train_dataset.W, train_dataset.D]

    if args.backbone == 'unet':
        net = Seg_Obelisk_Unet(num_labels, full_res)
        # optimizer and lr_scheduler initial after network initial
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    elif args.backbone == 'swinunet':
        net = SwinUNETR(img_size=full_res,
                        in_channels=1,
                        out_channels=num_labels,
                        feature_size=args.feature_size,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=args.use_checkpoint)
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)

    net.cuda()
    net.train()

    if args.pretrained:
        net.load_state_dict(torch.load(args.pretrained)["state_dict"], strict=False)
        logger.info(f"Model training resume from official model {args.pretrained}")
    if args.resume:
        ckpt = torch.load(d_options['resume'])
        net.load_state_dict(ckpt["state_dict"])
        if args.from_ckpt:
            best_acc = ckpt["best_acc"]
            star_epoch = ckpt["epoch"]
        logger.info(f"Model training resume from our model {args.resume}, and star from ckpt? {args.from_ckpt}.")
    else:
        net.apply(init_weights) if args.backbone == 'unet' else None

    if args.apply_lr_scheduler:
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, min_lr=0.00001, patience=10)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                    warmup_steps=d_options["warmup_steps"],
                                                    total_steps=end_epoch, )
        logger.info(f"Training with lr schedular warmup cosine")

    if args.from_ckpt:
        try:
            # there is no downstheres if resume from pretrained model
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt["scheduler"]) if args.apply_lr_scheduler else None
            logger.info(f"Optimizer and scheduler resume")
        except:
            pass

    ti_weight = d_options['ti_weight']
    ce_weight = d_options["ce_weight"]

    logger.info(f'obelisk params: {countParam(net)}')  # obelisk params 229217
    # logger.info(f'initial offset std: {torch.std(net.offset1.data).item() :.3f}')  # initial offset std 0.047

    run_loss = np.zeros([end_epoch, 3])

    dice_all_val = np.zeros((len(val_dataset), num_labels - 1))
    logger.info(f'Training set sizes: {len(train_dataset)}, Validation set sizes: {len(val_dataset)}')

    # for loop over iterations and epochs
    for epoch in range(star_epoch, end_epoch):
        run_loss[epoch] = 0.0

        t0 = time.time()

        for imgs, segs in train_loader:
            # img and seg have shape: [N, 1, H, W, D]
            if np.random.choice([0, 1]):
                # 50% to apply data augment
                with torch.no_grad():
                    imgs_cuda, y_label = augment_affine(imgs.cuda(), segs.cuda(), strength=0.075)
                    torch.cuda.empty_cache()
            else:
                imgs_cuda, y_label = imgs.cuda(), segs.cuda()

            optimizer.zero_grad()

            # forward path and loss
            predict = net(imgs_cuda)

            ce_loss = criterion(predict, y_label.squeeze(1))
            total_loss = ce_weight * ce_loss

            ti_loss = ti_criterion(predict, y_label) if ti_criterion else torch.tensor(0)
            total_loss += ti_weight * ti_loss
            run_loss[epoch, 1] += ti_weight * ti_loss.item()

            total_loss.backward()

            run_loss[epoch, 0] += total_loss.item()
            run_loss[epoch, 2] += ce_weight * ce_loss.item()

            optimizer.step()

            del total_loss
            del predict
            torch.cuda.empty_cache()
            del imgs_cuda
            del y_label
            torch.cuda.empty_cache()

        if args.apply_lr_scheduler:
            # scheduler.step(run_loss[epoch, 0])  # epoch wise lr decay
            scheduler.step()

        # evaluation on training images
        t1 = time.time() - t0

        if epoch % d_options['interval'] == 0:
            net.eval()

            for val_idx, (imgs, segs) in enumerate(val_loader):
                imgs_cuda = imgs.cuda()
                t0 = time.time()

                with torch.no_grad():
                    predict = net(imgs_cuda)
                    argmax = torch.argmax(predict, dim=1)
                    torch.cuda.synchronize()
                    time_i = (time.time() - t0)
                    dice_one_val = dice_coeff(argmax.cpu(), segs)
                dice_all_val[val_idx] = dice_one_val
                del predict
                del imgs_cuda
                torch.cuda.empty_cache()

            # logger some feedback information
            all_val_dice_avgs = dice_all_val.mean(axis=0)
            mean_all_dice = all_val_dice_avgs.mean()
            latest_lr = optimizer.state_dict()['param_groups'][0]['lr']

            is_best = mean_all_dice > best_acc
            best_acc = max(mean_all_dice, best_acc)

            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            logger.info(
                f"epoch {epoch}, time train {round(t1, 3)}, time infer {round(time_i, 3)}, "
                f"total loss {run_loss[epoch, 0] :.3f}, TI loss {run_loss[epoch, 1] :.3f}, "
                f"ce loss {run_loss[epoch, 2] :.3f} "
                f"dice_avgs {all_val_dice_avgs}, avgs {mean_all_dice :.3f}, "
                f"best_acc {best_acc :.3f}, lr {latest_lr :.8f}")

            if is_visdom:
                # loss line
                vis.line(Y=[run_loss[epoch]], X=[epoch], win='loss-', update='append', opts=loss_opts)
                # acc line
                vis.line(Y=[all_val_dice_avgs], X=[epoch], win='acc-', update='append', opts=acc_opts)
                vis.line(Y=[mean_all_dice], X=[epoch], win='best_acc-', update='append', opts=best_acc_opt)
                # lr decay line
                vis.line(Y=[latest_lr], X=[epoch], win='lr-', update='append', opts=lr_opts)

            net.cpu()

            state_dict = {
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if args.apply_lr_scheduler else None,
                "best_acc": best_acc,
                "epoch": epoch,
            }

            torch.save(state_dict, d_options['output'] + f"{d_options['dataset']}_latest.pth")

            if is_best:
                np.save(f"{d_options['output']}run_loss.npy", run_loss)
                torch.save(state_dict, d_options['output'] +
                           f"{d_options['dataset']}_best_{str(round(best_acc, 3)) if best_acc > 0.7 else ''}.pth")
                logger.info(f"saved the best model at epoch {epoch}, with best acc {best_acc :.3f}")
                if is_visdom:
                    vis.line(Y=[best_acc], X=[epoch], win='best_acc-', update='append', opts=best_acc_opt)

            net.cuda()
            net.train()


if __name__ == '__main__':
    main()
