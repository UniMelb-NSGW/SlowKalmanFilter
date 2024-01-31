import numba as nb
import numpy as np 

@nb.jit(nopython=True) #From https://stackoverflow.com/a/58017351
def block_diag_view_jit(arr, num):
    rows, cols = arr.shape
    result = np.zeros((num * rows, num * cols), dtype=arr.dtype)
    for k in range(num):
        result[k * rows:(k + 1) * rows, k * cols:(k + 1) * cols] = arr
    return result









"""
Given a Bilby dict, make it a numpy array 
"""
def dict_to_array(some_dict,target_keys):
    selected_dict = {k:some_dict[k] for k in target_keys}
    return np.array(list(selected_dict.values())).flatten()
