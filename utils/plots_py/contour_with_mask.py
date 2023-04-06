import os
import pathlib
import numpy as np
import cv2
from matplotlib import pyplot as plt


colors_tcia = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),  # 红、绿、蓝、黄
               (0, 255, 255), (255, 0, 255), (255, 140, 0)]

colors_lpba = [(0, 255, 0), (0, 255, 0), (0, 0, 255),
               (0, 0, 255), (255, 0, 0), (255, 0, 0)]


def draw_contour(img_path, mask_path, save_dir='./', save_name='gt_fixed.jpg', dataset="tcia", thickness=3, auto_save=False):
    if not os.path.exists(save_dir):
        # os.makedirs(out_dir, exist_ok=True)
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    if img is None or mask is None:
        raise ValueError("img_path or mask_path is wrong")
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # plt.imshow(mask)
    # plt.show()
    labels = np.unique(mask)
    print("labels:", labels)

    color_idx = 0
    for idx, i in enumerate(labels[1:]):  # 排除背景 0
        if dataset == 'lpba':
            if i not in [212, 217, 231, 236, 240, 245]: continue
            colors = colors_lpba
        if dataset == 'tcia':
            if i not in [31, 63, 95, 191, 223]: continue
            colors = colors_tcia
        if dataset == 'ct-mr':
            # if i not in [63, 127, 191, 255]: continue
            colors = colors_tcia

        label = np.where(mask != i, 0, i).astype(np.uint8)

        ## 辨认哪个器官的分割，因为二值化后的mask值不一样
        # plt.title(str(i))
        # plt.imshow(label)
        # plt.show()
        # plt.pause(3)

        ret, thresh = cv2.threshold(label, 0, 255, 0)
        # plt.imshow(thresh)
        # plt.show()

        # 第二个参数是轮廓检索模式，有 RETR_LIST, RETR_TREE, RETR_EXTERNAL, RETR_CCOMP
        contours, im = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 第一个参数是轮廓
        cv2.drawContours(image=img, contours=contours, contourIdx=-1,
                         color=colors[color_idx], thickness=thickness)  # thickness, 线的粗细
        color_idx += 1

    cv2.namedWindow('b', flags=cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL 窗口合适大小
    cv2.imshow('b', img)

    if auto_save:
        cv2.imwrite(os.path.join(save_dir, save_name), img)
        print(f"{save_name} saved to {save_dir}")
        cv2.destroyAllWindows()
    else:
        k = cv2.waitKey(0)  # 0，使窗口一直挂起
        if k == 27:  # 按下 esc 时，退出
            cv2.destroyAllWindows()
        elif k == ord('s'):  # 按下 s 键时保存并退出
            cv2.imwrite(os.path.join(save_dir, save_name), img)
            print(f"{save_name} saved to {save_dir}")
            cv2.destroyAllWindows()


if __name__ == "__main__":

    root = r"F:\shb_src\from_github\multi_modal_reg\output\SreenShot\reg\ct-mr"

    pref = r"swinU_785"
    for i in range(6, 18, 2):
        num = str(i).zfill(2)
        img_path = os.path.join(root, pref, f"img_tcia_{num}.jpg")
        if not os.path.isfile(img_path):
            continue
        mask_path = os.path.join(root, pref, f"seg_tcia_{num}.jpg")
        save_dir = os.path.join(root, pref, f"assets_line")
        save_name = os.path.split(mask_path)[-1]

        draw_contour(img_path, mask_path, save_dir=save_dir, save_name=save_name, dataset='ct-mr', thickness=10, auto_save=True)
