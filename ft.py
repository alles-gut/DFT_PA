##ft.py
##perform DFT and IDFT

import numpy as np

#perform fourier-transform
#arr  - array input
#size - after zerro padding size
def ft (arr, size):
    col_arr  = arr.shape[0]
    row_arr  = arr.shape[1]
    size_arr = arr.size

    if size[0] >= col_arr and size[1] >= row_arr:
        temp = np.zeros(size)
        temp[0:col_arr, 0:row_arr] = arr
        arr = temp
        col_arr = size[0]
        row_arr = size[1]
        size_arr = col_arr * row_arr

    m = np.ones([col_arr, col_arr])
    wm = np.ones([col_arr, col_arr])
    n = np.ones([row_arr, row_arr])
    wn = np.ones([row_arr, row_arr])
    for i in range(col_arr):
        m[i] = -2 * np.pi * m[i] * i / col_arr
        wm[:,i] = i
    for i in range(row_arr):
        n[:,i] = -2 * np.pi * n[:,i] * i / row_arr
        wn[i] = i
    wm = np.exp(wm * m * 1j)
    wn = np.exp(wn * n * 1j)

    ft_arr = np.zeros([col_arr, row_arr],dtype=complex)
    ft_arr = np.matmul(np.matmul(wm, arr), wn) / size_arr

    return ft_arr


#perform inverse fourier-transform
def invFt (arr):
    col_arr  = arr.shape[0]
    row_arr  = arr.shape[1]

    m = np.ones([col_arr, col_arr])
    wm = np.ones([col_arr, col_arr])
    n = np.ones([row_arr, row_arr])
    wn = np.ones([row_arr, row_arr])
    for i in range(col_arr):
        m[i] = 2 * np.pi * m[i] * i / col_arr
        wm[:,i] = i
    for i in range(row_arr):
        n[:,i] = 2 * np.pi * n[:,i] * i / row_arr
        wn[i] = i
    wm = np.exp(wm * m * 1j)
    wn = np.exp(wn * n * 1j)

    invft_arr = np.zeros([col_arr, row_arr],dtype=complex)
    invft_arr = np.matmul(np.matmul(wm, arr), wn)

    return invft_arr
