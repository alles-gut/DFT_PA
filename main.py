##main.py
##Jung, Hyunjun Programing Assignment
##EC3202 Signals and Systems
##2020 Spring Semester, GIST college


import cv2
import numpy as np
import random
import os

from dataio import *
from conv import *
from ft import *
from psnr import *
from noise import *
from filter import *


if __name__ == '__main__':

    img_path    = 'img.png'
    mat_path    = 'blur_kernel.mat'
    mat_feature = 'PSF'

    orig_img    = loadImage(img_path).astype('float64') / 255
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
    deblur = abs(deblur)
    print("Task 1 done")


    #Task 2
    print("\nTask 2")
    blur_img_noised = makeSomeNoise(0, 1e-3, blur_img)

    print("DFT blur_img_noised")
    ft_blur_img_noised = ft(blur_img_noised, blur_img.shape)
    print("IDFT performing - deblur noise/blurred image")

    sample_noise = makeSomeNoise(0, 1e-3, np.zeros(blur_img.shape))
    snr = abs(ft(blur_img_noised, blur_img_noised.shape))**2\
          /(np.sum(abs(ft(sample_noise, sample_noise.shape))**2)\
          /sample_noise.size)
    K = 1 / snr
    print(" with K :", K)
    noise_deblur = invFt(ft_blur_img_noised\
                         * np.conj(ft_blur_kernel)\
                         / (abs(ft_blur_kernel)**2 + K))\
                         [0:orig_img.shape[0], 0:orig_img.shape[1]]
    noise_deblur = abs(noise_deblur)


    #Task 3
    print("\nTask 3")
    psnr_task1 = psnr(orig_img, deblur)
    print("deblur without noise PSNR =", psnr_task1)
    psnr_task2 = psnr(orig_img, noise_deblur)
    print("deblur with noise PSNR =", psnr_task2)

    #Analize PSNR accourding to change of K
    print("apply K = K * 1000")
    K_ = K * 1000
    k_mult1000 = invFt(ft_blur_img_noised\
                       * np.conj(ft_blur_kernel)\
                       / (abs(ft_blur_kernel)**2 + K_))\
                       [0:orig_img.shape[0], 0:orig_img.shape[1]]
    k_mult1000 = abs(k_mult1000)

    print("apply K = K / 1000")
    K_ = K / 1000
    k_div1000  = invFt(ft_blur_img_noised\
                       * np.conj(ft_blur_kernel)\
                       / (abs(ft_blur_kernel)**2 + K_))\
                       [0:orig_img.shape[0], 0:orig_img.shape[1]]
    k_div1000 = abs(k_div1000)
    print(k_div1000)

    psnr_kmult1000 = psnr(orig_img, k_mult1000)
    psnr_kdiv1000  = psnr(orig_img, k_div1000)
    print("kmult1000 PSNR =", psnr_kmult1000)
    print("kdiv1000  PSNR =", psnr_kdiv1000)


    #Additional Task - improve deblurring quality
    print("\nAdditional Task")

    average_K  = filter(noise_deblur, 'average', 3, None)
    gaussian_K = filter(noise_deblur, 'gaussian', 3, 1)
    median_K   = filter(noise_deblur, 'median', 3, None)

    psnr_average_K  = psnr(orig_img, average_K)
    psnr_gaussian_K = psnr(orig_img, gaussian_K)
    psnr_median_K   = psnr(orig_img, median_K)

    print("average_K   PSNR =", psnr_average_K)
    print("gaussian_K  PSNR =", psnr_gaussian_K)
    print("median_K    PSNR =", psnr_median_K)

    K_ = 1 / snr * np.sqrt(1e-3)
    noise_deblur_k_ = invFt(ft_blur_img_noised\
                         * np.conj(ft_blur_kernel)\
                         / (abs(ft_blur_kernel)**2 + K_))\
                         [0:orig_img.shape[0], 0:orig_img.shape[1]]
    noise_deblur_k_ = abs(noise_deblur_k_)
    #noise_deblur_k_ = crimmins(abs(noise_deblur_k_)*255)/255

    average_K_  = filter(noise_deblur_k_, 'average', 3, None)
    gaussian_K_ = filter(noise_deblur_k_, 'gaussian', 3, 1)
    median_K_   = filter(noise_deblur_k_, 'median', 3, None)

    psnr_noise_deblur_k_ = psnr(orig_img, noise_deblur_k_)
    psnr_average_K_  = psnr(orig_img, average_K_)
    psnr_gaussian_K_ = psnr(orig_img, gaussian_K_)
    psnr_median_K_   = psnr(orig_img, median_K_)

    print("noise_K_    PSNR =", psnr_noise_deblur_k_)
    print("average_K_  PSNR =", psnr_average_K_)
    print("gaussian_K_ PSNR =", psnr_gaussian_K_)
    print("median_K_   PSNR =", psnr_median_K_)


    #show results
    ft_orig_img = ft(orig_img, blur_img.shape)
    ft_deblur = ft(deblur, blur_img.shape)
    ft_noise_deblur = ft(noise_deblur, blur_img.shape)
    ft_k_mult1000 = ft(k_mult1000, blur_img.shape)
    ft_k_div1000 = ft(k_div1000, blur_img.shape)


    #out the result
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
                              'Task3 - Apply K mul 1000',\
                              'Task3 - Apply K div 1000',\
                              'FT - Blur with noise',\
                              'FT - Noise deblur',\
                              'FT - deblur with K mul 1000',\
                              'FT - deblur with K div 1000',\
                              'average_k',\
                              'gaussian_k',\
                              'median_k',\
                              'noise_deblur_k_',\
                              'average_k_',\
                              'gaussian_k_',\
                              'median_k_']
    image_print_image_list = [ orig_img*255,\
                               blur_kernel/np.max(blur_kernel)*255,\
                               blur_img*255,\
                               deblur*255,\
                               abs(ft_orig_img),\
                               abs(ft_blur_kernel)*255,\
                               abs(ft_blur_img),\
                               abs(ft_deblur),\
                               blur_img_noised*255,\
                               noise_deblur*255,\
                               k_mult1000*255,\
                               k_div1000*255,\
                               abs(ft_blur_img_noised),\
                               abs(ft_noise_deblur),\
                               abs(ft_k_mult1000),\
                               abs(ft_k_div1000),\
                               average_K*255,\
                               gaussian_K*255,\
                               median_K*255,\
                               noise_deblur_k_*255,\
                               average_K_*255,\
                               gaussian_K_*255,\
                               median_K_*255]

    print("\nSaving result images\n")
    saveImage(image_print_name_list, \
              image_print_image_list)
    print("\nImage saved!\n")
