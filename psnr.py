##psnr.py
##calculate PSNR of img image for base image

import numpy as np

def psnr(base, obj):
    if base.shape != obj.shape :
        print("ERROR! shape of input images are not match")
        return -1

    R = 1
    mse = np.sum((base - obj) ** 2) / base.size
    psnr = 10 * np.log10((R**2)/mse)

    return psnr
