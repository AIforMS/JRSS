import os
from glob import glob
from PIL import Image


def crop_(src_file, dst_path):
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    file_name = os.path.split(src_file)[-1]
    img = Image.open(src_file)
    # 350,0 表示要裁剪的位置的左上角坐标，1380, 1296表示右下角, 通过画图打开查看坐标点
    region = img.crop((350, 0, 1380, 1296))
    region.save(os.path.join(dst_path, file_name))
    print("croped finished!")


if __name__ == '__main__':
    src_root = r"D:\code_sources\from_github\Medical Images Seg & Reg\MICCAI2020\seg-cross-reg\output\paper_assets\LPBA\seg\assets_line10\lpba_mr20"
    for src_file in glob(f"{src_root}/best_88.png"):
        if src_file is None:
            raise ValueError("glob nothing, mabey postfix is wrong")
        crop_(src_file, dst_path=os.path.join(src_root, "croped"))
