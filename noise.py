##noise.py
##make and return gaussian distribution noise

import numpy as np

def makeSomeNoise(mean, var, img):
    print("Generating noise")
    sigma = var**0.5
    shape = img.shape
    noise = np.random.normal(mean,sigma,shape)

    noisy_img = img + noise
    noisy_img[noisy_img>1] = 1
    noisy_img[noisy_img<0] = 0

    return noisy_img
