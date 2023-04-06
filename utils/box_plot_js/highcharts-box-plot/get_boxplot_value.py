import os
import numpy as np

"""
【函数说明】获取箱体图特征
【输入】 input_list 输入数据列表
【输出】 out_list：列表的特征[下限，Q1,Q2,Q3,上限] 和 err_point_list：异常值
【版本】 V1.0.0
【日期】 2023 03 23
"""


def box_feature(input_list):
    # 获取箱体图特征
    percentile = np.percentile(input_list, (25, 50, 75), interpolation='linear')
    # 以下为箱线图的五个特征值
    Q1 = np.around(percentile[0], 3)  # 上四分位数
    Q2 = np.around(percentile[1], 3)
    Q3 = np.around(percentile[2], 3)  # 下四分位数
    IQR = Q3 - Q1  # 四分位距
    ulim = Q3 + 1.5 * IQR  # 上限 非异常范围内的最大值
    llim = Q1 - 1.5 * IQR  # 下限 非异常范围内的最小值
    # llim = 0 if llim < 0 else llim
    # out_list = [llim,Q1,Q2,Q3,ulim]
    # 统计异常点个数
    # 正常数据列表
    right_list = []
    err_point_list = []
    value_total = 0
    average_num = 0
    for item in input_list:
        if item < llim or item > ulim:
            err_point_list.append(item)
        else:
            right_list.append(item)
            value_total += item
            average_num += 1
    average_value = value_total / average_num  # 平均值
    # 特征值保留一位小数
    out_list = [min(right_list), Q1, Q2, Q3, max(right_list)]
    return out_list, err_point_list, average_value


def get_4_ROIs(log_file=""):
    liver_list = []
    spleen_list = []
    r_kidney_list = []
    l_kidney_list = []

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if "[line:50]" in line:
                dsc = float(line.split(':')[-1])
                if dsc == 0:
                    continue
                if 'liver' in line:
                    liver_list.append(dsc)
                elif 'spleen' in line:
                    spleen_list.append(dsc)
                elif 'r_kidney' in line:
                    r_kidney_list.append(dsc)
                elif 'l_kidney' in line:
                    l_kidney_list.append(dsc)
    return liver_list, spleen_list, r_kidney_list, l_kidney_list


if __name__ == '__main__':
    # a baseline including below:
    root = r"F:/shb_src/from_github/multi_modal_reg/"

    # ct -> mr:
    swin_772 = r"output/reg_results/swin_unetr_ct2mr_0.772/test.log"
    swin_785 = r"output/reg_results/swin_unetr_ct2mr_ti0.785/test.log"
    swin_771 = r"output/reg_results/swin_unetr_ct2mr0.771/test.log"
    swin_757 = r"output/reg_results/swin_unetr_ct2mr0.757/test.log"
    deeds = r"output/reg_results/Deeds/ct-mr/dsc.log"

    # mr-> ct:
    swin_763 = r"output/reg_results/swin_unetr_mr2ct0.763/test.log"
    swin_754 = r"output/reg_results/swin_unetr_mr2ct0.754/test.log"
    swin_744 = r"output/reg_results/swin_unetr_mr2ct0.744/test.log"
    swin_737 = r"output/reg_results/swin_unetr_mr2ct0.737/test.log"
    deeds_ = r"output/reg_results/Deeds/mr-ct/dsc.log"

    liver, spleen, l_kidney, r_kidney = get_4_ROIs(log_file=os.path.join(root, swin_744))

    print("liver:", box_feature(liver)[0], ", error point:", box_feature(liver)[1])
    print("spleen:", box_feature(spleen)[0], ", error point:", box_feature(spleen)[1])
    print("l_kidney:", box_feature(l_kidney)[0], ", error point:", box_feature(l_kidney)[1])
    print("r_kidney:", box_feature(r_kidney)[0], ", error point:", box_feature(r_kidney)[1])
