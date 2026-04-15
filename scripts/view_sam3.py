import matplotlib
matplotlib.use('TkAgg')
import numpy as np, matplotlib.pyplot as plt

a = np.load('/home/chengzhe/Data/OMS_data3/rs_bags/20260127_015119_20260414_232749/sam3_alpha_cache/alpha_001409.npz')['alpha']
plt.imshow(a, cmap='hot', vmin=0, vmax=1)
plt.colorbar(); plt.title('SAM3 alpha (0=seam, 1=interior)')
plt.show()