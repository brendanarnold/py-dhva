import numpy as np
from fft_windows import nonzero_hann_window


def filter_fn(fx, rng, width):
    '''
    The filter from Tony's program ...

    filterfmini:=trunc(filterfmin*(nnpt*invbstep))*2;
    filterfmaxi:=trunc(filterfmax*(nnpt*invbstep))*2;
    filterwidthi:=(filterwidth*(nnpt*invbstep))*2;
    fftdata[i]:=fftdata[i]*(((tanh((i-filterfmini)/filterwidthi)+1)/2+(1-tanh((i-filterfmaxi)/filterwidthi))/2)-1)
    '''
    width = width / np.pi
    filter = 0.5 * ( np.tanh((fx - min(rng)) / width) + np.tanh(-(fx - max(rng)) / width) )
    return filter



def torque_filter(bvals, torque, filter_range, b_range=None, smoothing_width=100, fft_points=1024, fft_padding=4, window=nonzero_hann_window, background_poly=5, ret_invB=False, ret_all=False):
        '''
        Filters dHvA torque data similar to 'FFT Filter'
        '''
        # Doing a real FFT so need to double the points to emulate Tony's filter
        fft_points = 2 * fft_points
        if b_range is None: 
            b_range = [bvals.min(), bvals.max()]
        # Interpolate to evenly spaced data in 1/B
        inds = np.argsort(bvals) # x vals need to be increasing
        bvals = bvals[inds]
        torque = torque[inds]
        ibmin = 1.0 / max(b_range)
        ibmax = 1.0 / min(b_range)
        ibvals = np.linspace(ibmin, ibmax, fft_points)
        itorque = np.interp(1.0/ibvals, bvals, torque)
        # Remove background
        pcoeffs = []
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
        # Are dealing with real FFTs, need to shift the output of fftfreq
#        fx = np.fft.fftfreq(len(f), ib_int)
#        fx = (np.concatenate(( fx[len(fx)/2:], fx[:len(fx)/2] )) + fx.max()) / 2.0
        # Store a copy if full return dataset is specified
        if ret_all:
            f_nofilt = f.copy()
        # Apply filter
        filt = filter_fn(fx, filter_range, smoothing_width)
        f = f * filt
        # Bring back to reality
        itorque = np.fft.irfft(f)
        # Get rid of padding
        itorque = itorque[:fft_points]
        # Keep it real
#        itorque = itorque.real        
        # Unapply window
        # n.b. this causes edges to diverge as most functions go to zero at edges
        itorque = itorque / window(len(itorque))
        # Interpolate to even points in B if required
        if ret_invB:
            ft_bvals = ibvals
            ft_torque = itorque
        else:
            ft_bvals = np.linspace(b_range[0], b_range[1], fft_points)
            ft_torque = np.interp(ft_bvals, (1.0/ibvals)[::-1], itorque[::-1])
        if ret_all:
            ret_dict = {'band' :  filt,
                        'inv_b' : fx,
                        'inv_torque' : f_nofilt,
                        'smoothing_width' : smoothing_width,
                        'fft_points' : fft_points / 2,
                        'fft_padding' : fft_padding,
                        'window' : window(len(itorque)),
                        'poly_coeffs' : pcoeffs
            }
            return (ft_bvals, ft_torque, ret_dict)
        else:
            return (ft_bvals, ft_torque)


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    # Some tests
    # ==========================================================================
    # Try on a file that is just two sine waves at different frequencies. Filter 
    # one out, leaving a frequency 2500 sine wave of amplitude 1.0
    filter_range = [800, 1200]
    smoothing_width = 50
    filename = os.path.join(os.path.dirname(__file__), 'tests', 'sin1000,2500.txt')
    data = np.loadtxt(filename, delimiter='\t', skiprows=1)
    field = data[:,0]
    torque = data[:,1]
    (ft_bvals, ft_torque, d) = torque_filter(field, torque, filter_range, ret_all=True)
    # Compare with FFT FIlter program
    tony_filename = os.path.join(os.path.dirname(__file__), 'tests', 'tonyfilt_sin1000,2500_Hanning_band=0,8-1,2kT_smooth=50T.dat')    
    tony_data = np.loadtxt(tony_filename, delimiter='\t', skiprows=1)
    plt.figure()
    plt.plot(ft_bvals, ft_torque, color='blue')
    plt.plot(tony_data[:,0], tony_data[:,1], color='red')
    plt.show()
