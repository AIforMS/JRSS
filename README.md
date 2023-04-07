# JRSS
Code of [Shi, H., Lu, L., Yin, M., Zhong, C., & Yang, F. (2023). Joint few-shot registration and segmentation self-training of 3D medical images. Biomedical Signal Processing and Control, 80, 104294.](https://doi.org/10.1016/j.bspc.2022.104294)


## VS Code 远程链接超算

参考：[使用 VS Code + Remote-SSH 插件在本地跑超算](https://blog.csdn.net/qq_36484003/article/details/109595387)

注意 `~/.ssh/config` 文件中填入以下：

```shell script
# 超算 Remote-SSH
Host name-of-ssh-host-here  # 你在SSH远程主机上的用户名
    User your-user-name-on-host  # 你在SSH远程主机上的用户名
    HostName 8.8.8.8  # 远程超算的IP地址
    Port 8125  # 指定端口，否则默认端口为22
    IdentityFile ~/.ssh/id_rsa-remote-ssh  # 指定本地的私钥文件的路径
```

## 训练

使用 slurm 脚本在超算训练

```shell script
#!/bin/bash
#SBATCH -J mono-reg
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o logs/trainOut.txt
#SBATCH -e logs/trainErr.txt

SLURM_SUBMIT_DIR=~/JRSS
cd $SLURM_SUBMIT_DIR
NP=$SLURM_JOB_GPUS

CUDA_VISIBLE_DEVICES=$NP python train_mono_reg.py \
    # 使用 visualDL 进行可视化，运行之后终端打开服务，通过 VS Code 转发端口可以在本地浏览器访问
    -visualdl true \    
    -backbone swinunet \
    -dataset ct-mr \
    -batch_size 1 \
    -interval 1 \
    -lr 1e-4 \
    -weakly_sup \           # 使用分割标签进行弱监督
    -apply_ti_loss \        # 使用 TI Loss
    -use_checkpoint \       # swinunet作为backbone时可以节省内存
    -apply_lr_scheduler \   # lr warmup 并且按照余弦衰减
    -feature_size 24 \      # swinunet作为backbone时
    -epochs 1500 \
    -alpha 4.0 \
    -sim_weight 1.0 \
    -seg_weight 1.0 \
    -ti_weight 1e-5 \
    -sim_loss MIND \
    -output output/swinunet_mono/ \

    # 以下默认图像和分割标签在同一文件夹，并且文件名只有 img 和 seg 之差；
    # 若要定制，自行更改 `utils.datasets.py` 里的代码
    -mov_folder path/to/mov_folder \
    -mov_name ?_bcv_CT.nii.gz \
    -fix_folder path/to/fix_folder \
    -fix_name ?_chaos_MR.nii.gz \
    -train_pair_numbers "1 2 3 4 6 7 8 9" \
    -val_pair_numbers "5 10"

```

## 预测
可以本地运行，swinunetr做backbone需要10GB内存，显存不够可以在cpu上跑。

`register.py`

## 结果可视化

`utils/plots_py/contour_with_mask.py`，[Here](utils/plots_py/README.md)

## 箱线图绘制

`utils/box_plot_js/highcharts-box-plot/boxplots.html`，[Here](utils/box_plot_js/highcharts-box-plot/README.md)

