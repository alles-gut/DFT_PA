import numpy as np

#calculate PSNR of img image for base image
def psnr(base, obj):
    if base.shape != obj.shape :
        print("ERROR! shape of input images are not match")
        return -1

    mse = np.sum((base - obj) ** 2) / base.size
    psnr = 10 * np.log10(1/mse)

    return psnr
