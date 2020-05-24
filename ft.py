import numpy as np

#obtain fourier-transform. the imput arr has only real part
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

    m = np.ones([col_arr, row_arr])
    n = np.ones([col_arr, row_arr])
    for i in range(col_arr):
        m[i] = -2 * np.pi * m[i] * i / col_arr
    for i in range(row_arr):
        n[:,i] = -2 * np.pi * n[:,i] * i / row_arr

    ft_arr = np.zeros([col_arr, row_arr],dtype=complex)
    for u in range(int(col_arr)):
        for v in range(int(row_arr)):
            if u % 50 == 0 and v == 0:
                print("iteration :",u, v)
            ft_arr[u,v] = np.sum( arr * np.exp( ( m * u + n * v ) * 1j )) / size_arr

    return ft_arr


#perform inverse fourier-transform
def invFt (arr):
    col_arr  = arr.shape[0]
    row_arr  = arr.shape[1]
    size_arr = arr.size

    m = np.ones([col_arr, row_arr])
    n = np.ones([col_arr, row_arr])
    for i in range(col_arr):
        m[i] = 2 * np.pi * m[i] * i / col_arr
    for i in range(row_arr):
        n[:,i] = 2 * np.pi * n[:,i] * i / row_arr

    invft_arr = np.zeros([col_arr, row_arr],dtype=complex)
    for u in range(int(col_arr)):
        for v in range(int(row_arr)):
            if u % 50 == 0 and v == 0:
                print("Iteration :", u, v)
            invft_arr[u,v] = np.sum( arr * np.exp( ( m * u + n * v ) * 1j ))

    return invft_arr
