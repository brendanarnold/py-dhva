# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 20:34:27 2011

@author: ba1224

Filters dHvA torque data similar to 'FFT Filter'
        
  torque_fft(bvals, torque, b_range=None, fft_points=1024, fft_padding=4, window=nonzero_hann_window, background_poly=5, is_invB=False, ret_complex=False) 

"""

import numpy as np
from fft_windows import nonzero_hann_window

def torque_fft(bvals, torque, b_range=None, fft_points=1024, fft_padding=4, window=nonzero_hann_window, background_poly=5, is_invB=False, ret_complex=False):
        '''
        Filters dHvA torque data similar to 'FFT Filter'
        
        torque_fft(bvals, torque, b_range=None, fft_points=1024, fft_padding=4, window=nonzero_hann_window, background_poly=5, is_invB=False, ret_complex=False)        
        
        '''
        # Doing a real FFT so need to double the points to emulate Tony's filter
        fft_points = 2 * fft_points
        if b_range is None: 
            b_range = [bvals.min(), bvals.max()]
        if not is_invB:
            # Interpolate to evenly spaced data in 1/B
            inds = np.argsort(bvals) # x vals need to be increasing
            bvals = bvals[inds]
            torque = torque[inds]
            ibmin = 1.0 / max(b_range)
            ibmax = 1.0 / min(b_range)
            ibvals = np.linspace(ibmin, ibmax, fft_points)
            itorque = np.interp(1.0/ibvals, bvals, torque)
        else:
            ibvals = bvals
            itorque = torque
        # Remove background
        if background_poly is not None:
            pcoeffs = np.polyfit(ibvals, itorque, background_poly)
            itorque = itorque - np.polyval(pcoeffs, ibvals)
        # Apply window
        itorque = itorque * window(len(itorque))
        # Pad with zeros for finer FFT
        itorque = np.concatenate((itorque, np.zeros((fft_padding - 1) * fft_points)))
        # Do FFT
        f = np.fft.rfft(itorque)
        ib_int = (ibmax - ibmin) / fft_points
        fx = np.linspace(0, 0.5 / ib_int, len(f)) # FFT limits are the Nyquist freq
        # Return the complex FFT only if asked for
        if not ret_complex:
            f = np.abs(f)
        return (fx, f)
        
if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    # Some tests
    # ==========================================================================
    # Try on a file that is just two sine waves at different frequencies (100 and 2500)
    filename = os.path.join(os.path.dirname(__file__), 'tests', 'sin1000,2500.txt')
    data = np.loadtxt(filename, delimiter='\t', skiprows=1)
    field = data[:,0]
    torque = data[:,1]
    fx, f = torque_fft(field, torque, background_poly=1)
    plt.figure()
    plt.plot(fx, f, color='blue')
    plt.show()

          
        