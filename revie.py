import numpy as np


npy = np.load("./data/pseudo_labels/gt_binary.npy", allow_pickle=True)
nanlist = np.load("./data/pseudo_labels/nalist.npy", allow_pickle=True)
print(nanlist.shape, nanlist[-10:])
print(npy.shape, npy[-1000:], sum (npy))