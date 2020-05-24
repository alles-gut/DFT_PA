import cv2
import numpy as np
import random

from dataio import *
from conv import *
from ft import *

def generateNoise(size):
    print("Generating noise")
    noise = np.zeros(size)
    for i in range(size[0]):
        for i in range(size[1]):
            noise[i,j] = random.randint(0,255)
    return noise/255

if __name__ == '__main__':

    img_path    = 'img.png'
    mat_path    = 'blur_kernel.mat'
    mat_feature = 'PSF'

    orig_img    = loadImage(img_path)
    blur_kernel = loadmat(mat_path, mat_feature)


    #Task 1
    print("Task 1")
    print("make blur image using convolution")
    blur_img = basicConv(orig_img, blur_kernel) / 255

    print("DFT blur kernel")
    ft_blur_kernel = ft(blur_kernel, blur_img.shape)
    print("DFT blurred image")
    ft_blur_img    = ft(blur_img, blur_img.shape)
    print("IDFT performing")
    guess_original = invFt(ft_blur_img / ft_blur_kernel)[0:orig_img.shape[0], 0:orig_img.shape[1]]
    guess_original = abs(guess_original) / (np.sum(blur_img)/blur_img.size) / 255
    print("Task 1 done")

    #Task 2
    print("Task 2")
    noise = generateNoise([blur_img.shape])
    ft_noise = ft(noise, blur_img.shape)
    ft_orig_img = ft(orig_img, blur_img.shape)
    blur_img_noised = invFt(ft_orig_img * ft_blur_kernel + ft_noise)
 
    #show results
    image_print_name_list  = ['Original Image', 'Task1 - Blurred Image', 'Task1 - Guess Original']
    image_print_image_list = [ orig_img, blur_img, guess_original]
    showImage(image_print_name_list, \
            image_print_image_list)
