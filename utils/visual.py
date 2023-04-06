import os
import pathlib

import visdom
from visualdl import LogWriter


def visualdl_scalar_lines(args, epoch, losses, accs, best_acc, lr, accs_bi=None):
    """
    Doc: https://gitee.com/paddlepaddle/VisualDL/tree/develop/docs/components#visualdl-%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97
    URL: http://localhost:8040/app/scalar
    Params:
        :args: args
        :epoch: epoch
        :losses: run loss
        :accs, accs_bi: validation dsc list
        :best_acc: beast_acc
        :lr: latest lr with lr decay 
    """
    logdir = os.path.join(args.output, "scalar_lines")
    if not os.path.exists(logdir):
        pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)

    ROIs = ['spleen', 'pancreas', 'kidney', 'gallbladder', 'esophagus', 'liver',
            'stomach', 'duodenum'] if args.dataset == 'tcia' \
        else ['liver', 'spleen', 'r_kidney', 'l_kidney']

    with LogWriter(logdir) as writer:
        # losses
        writer.add_scalar(tag="loss/total_loss", step=epoch, value=losses[epoch, 0])
        writer.add_scalar(tag="loss/sim_loss", step=epoch, value=losses[epoch, 1])
        writer.add_scalar(tag="loss/grad_loss", step=epoch, value=losses[epoch, 2])
        writer.add_scalar(tag="loss/seg_loss", step=epoch, value=losses[epoch, 3])
        writer.add_scalar(tag="loss/ti_loss", step=epoch, value=losses[epoch, 4])
        # accs
        for idx, label in enumerate(ROIs):
            writer.add_scalar(tag=f"acc/{label}", step=epoch, value=accs[idx])
        writer.add_scalar(tag="acc/mean", step=epoch, value=accs.mean())
        if args.bidir:
            writer.add_scalar(tag="acc/mean_bi", step=epoch, value=accs_bi.mean())
        writer.add_scalar(tag="acc/best", step=epoch, value=best_acc)
        # lr decay
        writer.add_scalar(tag="lr/lr", step=epoch, value=lr)


def visualdl_images(args, step, seg_moved, seg_fixed):
    """
    URL: http://localhost:8040/app/sample/image
    Params:
        :args: args
        :step: validation idx
        :seg_moved: seg_moved tensor
        :seg_fixed: seg_fixed tensor
    """
    logdir = os.path.join(args.output, "image_visual")
    if not os.path.exists(logdir):
        pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)

    seg_m_np = seg_moved[0].long().cpu().squeeze().numpy()
    seg_f_np = seg_fixed[0].long().cpu().squeeze().numpy()

    scale = 255 // seg_f_np.max()
    seg_slice_m = seg_m_np[:, [68], :].T.transpose((0, 2, 1)).repeat(3, axis=2) * scale  # 冠状面
    seg_slice_f = seg_f_np[:, [68], :].T.transpose((0, 2, 1)).repeat(3, axis=2) * scale  # 冠状面

    with LogWriter(logdir) as writer:
        writer.add_image(tag="seg/moved", step=step, img=seg_slice_m, dataformats="HWC")
        writer.add_image(tag="seg/fixed", step=step, img=seg_slice_f, dataformats="HWC")


"""
for visdom
"""

def visdom_scalar_lines(args, epoch, losses, accs, lr):
    """
    URL: http://localhost:8097
    Params:
        :epoch: epoch
        :losses: run loss
        :accs: validation dsc list
        :lr: latest lr with lr decay 
    """
    vis = visdom.Visdom()

    loss_opts = {'xlabel': 'epochs',
                 'ylabel': 'loss',
                 'title': 'Loss Line',
                 'legend': ['total loss', 'sim loss', 'grad loss', 'dice loss', 'ti loss']}
    acc_opts = {'xlabel': 'epochs',
                'ylabel': 'acc',
                'title': 'Acc Line',
                'legend': ['1 spleen', '2 pancreas', '3 kidney', '4 gallbladder', '5 esophagus', '6 liver',
                           '7 stomach', '8 duodenum'] if args.dataset == 'tcia'
                else ['1 liver', '2 spleen', '3 right kidney', '4 left kidney']}
    lr_opts = {'xlabel': 'epochs', 'ylabel': 'lr', 'title': 'LR Line'}
    best_acc_opt = {'xlabel': 'epochs', 'ylabel': 'best acc', 'title': 'Best Acc Line'}
    # loss line
    vis.line(Y=[losses[epoch]], X=[epoch], win='loss', update='append', opts=loss_opts)
    # acc line
    vis.line(Y=[accs], X=[epoch], win='acc', update='append', opts=acc_opts)
    vis.line(Y=[accs.mean()], X=[epoch], win='mean-acc', update='append', opts=best_acc_opt)
    # lr decay line
    vis.line(Y=[lr], X=[epoch], win='lr', update='append', opts=lr_opts)


def visdom_images(args, seg_moved, seg_fixed):
    """
    URL: http://localhost:8097
    Params:
        :seg_moved: seg_moved tensor
        :seg_fixed: seg_fixed tensor
    """
    vis = visdom.Visdom()

    seg_m_np = seg_moved[0].int().cpu()
    seg_f_np = seg_fixed[0].int().cpu()

    scale = 255 // seg_f_np.max()
    seg_slice_m = seg_m_np[:, [68], :].T.permute(0, 2, 1).repeat(3, 1, 1) * scale  # 冠状面
    seg_slice_f = seg_f_np[:, [68], :].T.permute(0, 2, 1).repeat(3, 1, 1) * scale  # 冠状面

    vis.images(seg_slice_m, nrow=1, win='moved',
               opts={'title': 'Moving Images',
                     'width': 100, 'height': 500})
    vis.images(seg_slice_f, nrow=1, win='fixed',
               opts={'title': 'Moved Images',
                     'width': 100, 'height': 500})
