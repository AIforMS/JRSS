from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator


def main():
    # 把x轴的刻度间隔设置为 0.1，并存在变量里
    x_major_locator = MultipleLocator(0.1)
    # 把x轴的刻度间隔设置为 0.05，并存在变量里
    y_major_locator = MultipleLocator(0.1)

    t = [0.5, 0.6, 0.7, 0.8]
    dsc = {
        'n1': {
            'lpba': [0.61, 0.755, 0.761, 0.632],
            'tcia': [0.53, 0.69, 0.64, 0.545]
        },
        'n3': {
            'lpba': [0.695, 0.758, 0.769, 0.73],
            'tcia': [0.60, 0.717, 0.68, 0.667]
        },
        'n5': {
            'lpba': [0.71, 0.783, 0.791, 0.757],
            'tcia': [0.67, 0.737, 0.727, 0.697]
        },
        'n7': {
            'lpba': [0.758, 0.797, 0.808, 0.777],
            'tcia': [0.698, 0.77, 0.767, 0.717]
        },
    }

    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.9, hspace=None)
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15, 3))  # dpi=600
    # fig.subplots_adjust(hspace=0, wspace=0.3)

    for i, ax in zip([1, 3, 5, 7], axs):
        ax.spines['top'].set_visible(False)  # 不显示图表框的上边框
        ax.spines['right'].set_visible(False)  # 不显示图表框的右边框

        # ax.set_xlim(0.45, 0.85)
        ax.set_ylim(0.45, 0.85)

        # 轴的刻度
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)

        dsc_d = dsc[f'n{i}']

        ax.plot(t, dsc_d['lpba'], label='LPBA40', marker='D')
        ax.plot(t, dsc_d['tcia'], label='TCIA', marker='^')
        ax.set_title(f'N = {i}', fontsize=10)

        ax.set_xlabel('τ', fontsize=14)
        ax.set_ylabel('Seg-DSC') if i == 1 else None
        ax.legend(fontsize=10)
        ax.grid(axis='y', linewidth=0.3)

    fig.tight_layout()

    # plt.savefig(r'D:\code_sources\from_github\Medical Images Seg & Reg\MICCAI2020\seg-cross-reg\output\paper_assets\LPBA\boxplot\t_1357.jpg',
    #             dpi=600, format='jpg')
    plt.show()


if __name__ == '__main__':
    main()
