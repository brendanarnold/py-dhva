import numpy as np

def hann_window(len_vals):
    n = np.arange(len_vals)
    return 0.5 * (1 - np.cos(2 * np.pi * n / (len_vals - 1)))
    
def nonzero_hann_window(len_vals):
    '''A Hann window that does not go to zero, hence is ok for dividing'''
    n = np.linspace(1, len_vals - 2, len_vals)
    return 0.5 * (1 - np.cos(2 * np.pi * n / (len_vals - 1)))

