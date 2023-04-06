import numpy as np
import matplotlib.pyplot as plt

data_path = r"D:\code_sources\from_github\Medical Images Seg & Reg\multi-modal\multi_modal_reg\output\unet_no_ti\run_loss.npy"

loss_data = np.load(data_path)
total_loss = loss_data[:, 0]
ti_loss = loss_data[:, 1]  # here no ti loss data
ohem_loss = loss_data[:, 2]

x_data = np.arange(loss_data.shape[0])
plt.plot(x_data, loss_data[:, 2])
plt.show()
