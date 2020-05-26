import cv2
import numpy as np
import random

from dataio import *
from conv import *
from ft import *
from psnr import *


def generateNoise(mean, var, size):
    print("Generating noise")
    noise = np.zeros(size)
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,size)
    return gauss/4


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
    ft_blur_img = ft(blur_img, blur_img.shape)
    print("IDFT performing - ft_blur_img / ft_blur_kernel")
    deblur = invFt(ft_blur_img / ft_blur_kernel)[0:orig_img.shape[0], 0:orig_img.shape[1]]
    deblur = abs(deblur) / blur_img.size / 255
    print("Task 1 done")


    #Task 2
    print("Task 2")
    noise = generateNoise(0, 1e-3, blur_img.size)
    blur_img_noised = blur_img + noise

    print("DFT blur_img_noised")
    ft_blur_img_noised = ft(blur_img_noised, blur_img.shape)
    print("IDFT performing - deblur noise/blurred image")
    K = 
    noise_deblur = invFt(ft_blur_img_noised / (ft_blur_kernel + K))
    noise_deblur = invFt((abs(ft_blur_kernel) > 6*1e-8)*ft_blur_img_noised/ft_blur_kernel)
    noise_deblur = abs(inv)[0:img.shape[0], 0:img.shape[1]]/blur_img.size
 

    #Task 3
    print("Task 3")
    psnr_task1 = psnr(img, deblur)
    print("deblur without noise PSNR =", psnr_task1)
    psnr_task2 = psnr(img, noise_deblur)
    print("deblur with noise PSNR =", psnr_task2)


    #Additional Task - improve deblurring quality
    print("Additional Task")


    #show results
    image_print_name_list  = ['Original Image', 'Task1 - Blurred Image', 'Task1 - Guess Original', ]
    image_print_image_list = [ orig_img, blur_img, deblur, blur_img_noised]
    showImage(image_print_name_list, \
            image_print_image_list)
