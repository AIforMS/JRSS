from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator


def main():

    N = ['0', '1', '3', '5', '7', 'K']
    dsc = {
        'lpba40': {
            'seg': [None, 0.756, 0.765, 0.795, 0.805, 0.834],
            'reg': [0.71, 0.741, 0.751, 0.759, 0.766, 0.789]
        },
        'tcia': {
            'seg': [None, 0.691, 0.719, 0.753, 0.769, 0.800],
            'reg': [0.44, 0.514, 0.520, 0.539, 0.547, 0.555]
        },
    }

    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.9, hspace=None)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))  # dpi=600
    # fig.subplots_adjust(hspace=0, wspace=0.3)

    for key, ax in zip(dsc.keys(), axs):
        ax.spines['top'].set_visible(False)  # 不显示图表框的上边框
        ax.spines['right'].set_visible(False)  # 不显示图表框的右边框

        if key == 'lpba40':
            ax.set_ylim(0.7, 0.9)

        # # 轴的刻度
        # ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))

        base_line0, = ax.plot(N[:2], dsc[key]['seg'][:2], linestyle='--', marker='.')
        ax.plot(N[1:-1], dsc[key]['seg'][1:-1], label='Segmentation', marker='D', c=base_line0.get_color())
        # print(f"baseline: {base_line}")
        ax.plot(N[-2:], dsc[key]['seg'][-2:], marker='*', linestyle='--', c=base_line0.get_color())  # markersize=10, 

        base_line1, = ax.plot(N[:2], dsc[key]['reg'][:2], linestyle='--', marker='.')
        ax.plot(N[1:-1], dsc[key]['reg'][1:-1], marker='^', label='Registration', c=base_line1.get_color())
        ax.plot(N[-2:], dsc[key]['reg'][-2:], marker='*', linestyle='--', c=base_line1.get_color())  # markersize=10, 
        
        ax.set_title(f'{key.upper()}', fontsize=10)
        ax.set_xlabel('N')
        ax.set_ylabel('DSC') if key == 'lpba40' else None
        ax.legend(fontsize=10)
        ax.grid(axis='y', linewidth=0.3)

    fig.tight_layout()

    # plt.savefig(r'D:\code_sources\from_github\Medical Images Seg & Reg\MICCAI2020\seg-cross-reg\output\paper_assets\LPBA\boxplot\N_1357.svg',
    #             format='svg')  # dpi=600, 
    plt.show()



if __name__ == '__main__':
    main()
