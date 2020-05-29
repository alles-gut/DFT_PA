##main.py
##Jung, Hyunjun Programing Assignment
##EC3202 Signals and Systems
##2020 Spring Semester, GIST college


import cv2
import numpy as np
import random

from dataio import *
from conv import *
from ft import *
from psnr import *
from noise import *


if __name__ == '__main__':

    img_path    = 'img.png'
    mat_path    = 'blur_kernel.mat'
    mat_feature = 'PSF'

    orig_img    = loadImage(img_path) / 255
    blur_kernel = loadmat(mat_path, mat_feature)


    #Task 1
    print("\nTask 1")
    print("make blur image using convolution")
    blur_img = conv(orig_img, blur_kernel)

    print("DFT blur kernel")
    ft_blur_kernel = ft(blur_kernel, blur_img.shape)
    print("DFT blurred image")
    ft_blur_img = ft(blur_img, blur_img.shape)
    print("IDFT performing - ft_blur_img / ft_blur_kernel")
    deblur = invFt(ft_blur_img / ft_blur_kernel)\
            [0:orig_img.shape[0], 0:orig_img.shape[1]]
    deblur = abs(deblur) / blur_img.size
    print("Task 1 done")


    #Task 2
    print("\nTask 2")
    noise = makeSomeNoise(0, 1e-3, blur_img.shape)
    blur_img_noised = blur_img + noise

    print("DFT blur_img_noised")
    ft_blur_img_noised = ft(blur_img_noised, blur_img.shape)
    print("IDFT performing - deblur noise/blurred image")

    ft_noise = ft(noise, blur_img.shape)
    print(blur_img_noised)
    mu_sig = np.mean(blur_img_noised)
    print(mu_sig)
    sigma_sig = np.sqrt(np.sum((blur_img_noised - mu_sig)**2)/np.size(blur_img_noised))
    print('SNR =', mu_sig / sigma_sig)

    snr = np.sqrt(np.std(ft_blur_img_noised)**2/np.std(ft_noise)**2-1)
    print('SNR_ = ', snr)

    K = 2*1e-6
    print("  K :", K)
    noise_deblur = invFt(ft_blur_img_noised\
                 / (ft_blur_kernel + K))\
                 [0:orig_img.shape[0], 0:orig_img.shape[1]]
    noise_deblur = abs(noise_deblur) / blur_img.size


    #Task 3
    print("\nTask 3")
    psnr_task1 = psnr(orig_img, deblur)
    print("deblur without noise PSNR =", psnr_task1)
    psnr_task2 = psnr(orig_img, noise_deblur)
    print("deblur with noise PSNR =", psnr_task2)

    #Analize PSNR accourding to change of K
    print("apply K = K * 10")
    K_ = K * 10
    k_mult10 = invFt(ft_blur_img_noised\
            / (ft_blur_kernel + K_))\
            [0:orig_img.shape[0], 0:orig_img.shape[1]]
    k_mult10 = abs(k_mult10) / blur_img.size
    print("apply K = K / 10")
    K_ = K / 10
    k_div10 = invFt(ft_blur_img_noised\
            / (ft_blur_kernel + K_))\
            [0:orig_img.shape[0], 0:orig_img.shape[1]]
    k_div10 = abs(k_div10) / blur_img.size

    psnr_kmult10 = psnr(orig_img, k_mult10)
    psnr_kdiv10 = psnr(orig_img, k_div10)
    print("kmult10 PSNR =", psnr_kmult10)
    print("kdiv10 PSNR =", psnr_kdiv10)

    #Additional Task - improve deblurring quality
    print("Additional Task")


    #show results
    ft_orig_img = ft(orig_img, blur_img.shape)
    ft_deblur = ft(deblur, blur_img.shape)
    ft_noise_deblur = ft(noise_deblur, blur_img.shape)
    ft_k_mult10 = ft(k_mult10, blur_img.shape)
    ft_k_div10 = ft(k_div10, blur_img.shape)
    image_print_name_list  = ['Original Image',\
                              'Blur kernel',\
                              'Task1 - Blurred Image',\
                              'Task1 - Deblur',\
                              'FT - Original Image',\
                              'FT - Blur kernel',\
                              'FT - Blurred Image',\
                              'FT - Deblur',\
                              'Task2 - Blur with noise',\
                              'Task2 - Noise deblur',\
                              'Task3 - Apply K * 10',\
                              'Task3 - Apply K / 10',\
                              'FT - Blur with noise',\
                              'FT - Noise deblur',\
                              'FT - deblur with K * 10',\
                              'FT - deblur with K / 10']
    image_print_image_list = [ orig_img,\
                               blur_kernel/np.max(blur_kernel),\
                               blur_img,\
                               deblur,\
                               abs(ft_orig_img)*blur_img.size/255*2,\
                               abs(ft_blur_kernel)*blur_img.size,\
                               abs(ft_blur_img)*blur_img.size/255*2,\
                               abs(ft_deblur)*blur_img.size/255*2,\
                               blur_img_noised,\
                               noise_deblur,\
                               k_mult10,\
                               k_div10,\
                               abs(ft_blur_img_noised)*blur_img.size/255*2,\
                               abs(ft_noise_deblur)*blur_img.size/255*2,\
                               abs(ft_k_mult10)*blur_img.size/255*2,\
                               abs(ft_k_div10)*blur_img.size/255*2]

    print("show result image")
    showImage(image_print_name_list, \
              image_print_image_list)