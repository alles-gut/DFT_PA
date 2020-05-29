##noise.py
##make and return gaussian distribution noise

import numpy as np

def makeSomeNoise(mean, var, size):
    print("Generating noise")
    noise = np.zeros(size)
    sigma = var**0.5
    gauss_noise = np.random.normal(mean,sigma,size)

    return gauss_noise
