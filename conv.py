##conv.py
##convolution two array -  arr1 convolutioin arr2

import numpy as np

#Returns an array of the same shape as the zero padded input arr1
def conv(arr1, arr2):
    col_arr1 = arr1.shape[0]
    row_arr1 = arr1.shape[1]
    col_arr2 = arr2.shape[0] #should be odd number
    row_arr2 = arr2.shape[1] #should be odd number

    temp = arr1
    arr1 = np.zeros((col_arr1+col_arr2-1, row_arr1+row_arr2-1))
    arr1[int((col_arr2-1)/2):int((col_arr2-1)/2+col_arr1), \
            int((row_arr2-1)/2):int((row_arr2-1)/2+row_arr1)] \
            = temp
    del(temp)

    conv_arr = np.zeros((col_arr1+col_arr2-1, row_arr1+row_arr2-1))
    for i in range(col_arr1):
        for j in range(row_arr1):
            conv_arr[i:i+col_arr2, j:j+row_arr2]\
                    = conv_arr[i:i+col_arr2, j:j+row_arr2]\
                    + ( arr1[i+int((col_arr2-1)/2), j+int((row_arr2-1)/2)]\
                    * arr2 )

    return conv_arr
