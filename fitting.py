import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import linregress
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
import h5py
import matplotlib.gridspec as gridspec

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# %set_env MANPATH=/home/paul.torrey/local/texlive/2018/texmf-dist/doc/man:$MANPATH
# %set_env INFOPATH=/home/paul.torrey/local/texlive/2018/texmf-dist/doc/info:$INFOPATH
# %set_env PATH=/home/paul.torrey/local/texlive/2018/bin/x86_64-linux:/home/paul.torrey/local/texlive/2018/texmf-dist:$PATH
        
mpl.rc('font',**{'family':'sans-serif','serif':['Computer Modern'],'size':15})
mpl.rc('text', usetex=True)
mpl.rcParams['axes.linewidth'] = 2

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    '''Savitzky-Golay fitting function
    
    Directly from scipy cookbook: https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    '''
    from math import factorial
    
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def find_closest(rs,grad,value,tol=2e-3):
    '''Find the closest value to desired value (within some tolerance)
    
    '''
    lookAt = rs[(np.abs(grad - value) < tol) ]
    if len(lookAt) == 0:
        return np.nan
    breakidx = [int(np.where(rs == lookAt[i])[0]) for i in range(len(lookAt))]
    for idx in breakidx:
        if (grad[idx - 1] < grad[idx]):
            return rs[idx]
    return np.nan


def adapted_spline(rs,mprof,window=55,grad_val=-0.02,sf=0.1):
    '''Combine raw data into smoothed savitzky-golay and univariate spline fit
    
    '''
    mprof = np.array(mprof)
    
    rs    =    rs[(~np.isnan(mprof))] 
    mprof = mprof[(~np.isnan(mprof))]
    
    mprof = mprof[np.argsort(rs)]
    rs    =    rs[np.argsort(rs)]
    
    ### Profiles
    
    smoothed = savitzky_golay(mprof, window_size=window, order = 7) # order 7 is arbitrary pick    
    spl = UnivariateSpline(rs,smoothed)    
    spl.set_smoothing_factor(sf)


    ### Gradient 
    splgrad = spl.derivative(1)
    
    ### Break Radius
    break_val = find_closest(rs,splgrad(rs),grad_val)
            
    return spl(rs), splgrad(rs), break_val, rs, mprof

def stable_break(rsfrs,mprofs,grads,window_start=9,sf=0.1):
    '''Return the break radius
    
    'stable' refers to the window size -- if window size doesn't impact answer
    within standard deviation of 0.5 then the break radius is located
    
    '''
    keepGoing = True
    window_mid = window_start + 4
    spl2Use = None
    splgrad2use = None
    br2Use = None
    rs2Use = None
    mp2Use = None
    while keepGoing:
        spl1,splgrad1,br1,rs1,mprof1 = adapted_spline(rsfrs,mprofs,grad_val=grads,window=window_mid-4,sf=sf)
        spl2,splgrad2,br2,rs2,mprof2 = adapted_spline(rsfrs,mprofs,grad_val=grads,window=window_mid-2,sf=sf)
        spl3,splgrad3,br3,rs3,mprof3 = adapted_spline(rsfrs,mprofs,grad_val=grads,window=window_mid,sf=sf)
        spl4,splgrad4,br4,rs4,mprof4 = adapted_spline(rsfrs,mprofs,grad_val=grads,window=window_mid+2,sf=sf)
        spl5,splgrad5,br5,rs5,mprof5 = adapted_spline(rsfrs,mprofs,grad_val=grads,window=window_mid+4,sf=sf)
                
        breaks = np.array([br1,br2,br3,br4,br5])
                
        if len(breaks[(np.isnan(breaks))]) != 0:
            window_mid += 2
        else:
            median = np.nanmedian(breaks)

            std = np.nanstd(breaks)

            if std > 0.05:
                window_mid += 2
            else:
                keepGoing = False
                spl2Use = spl3
                splgrad2Use = splgrad3
                br2Use = br3
                rs2Use = rs3
                mp2Use = mprof3
            
        if window_mid > 100: # If the window size gets > 10 kpc -- only happens for idv profiles
            keepGoing = False
            window_mid = np.nan
            spl2Use = np.nan
            splgrad2Use = np.nan
            br2Use = np.nan
            rs2Use = np.nan
            mp2Use = np.nan
            print('Over 10.0 kpc, returning NaN')
    return spl2Use,splgrad2Use,br2Use,rs2Use,mp2Use,window_mid


########### Example call (stacked profiles) ###########

Type1 = 'RSFR'
Type2 = 'RSHM'

med_info  = h5py.File('Illustris_z0med.h5' ,'r') # File with median profile data 
rsfrFile  = h5py.File('Illustris_z0%s.h5'%Type1,'r') # File with RSFR data
rshmFile  = h5py.File('Illustris_z0%s.h5'%Type2,'r') # File with RSHM data

brs = []
rshms = []
rsfrs = []

currentMass = 8.5

for mbin in med_info:
    current = med_info.get(mbin)
    medprof = np.array(current.get('totmed'))
    mederr  = np.array(current.get('toterr'))

    rsfr = float(np.array(rsfrFile.get(mbin).get('median_rsfr50')))
    rsfrs.append(rsfr)

    rshm   = float(np.array(rshmFile.get(mbin).get('median_rshm')))
    rshms.append(rshm)

    rs = np.arange(0,len(mederr)/10,0.1)

    rins    = np.array(rin.get(mbin).get('median_rsfr10'))

    medprof = medprof[(rs > rins)]
    rs      =      rs[(rs > rins)]

    C = 0.28
    DIVISOR = 3.5
    C5 = C / DIVISOR

    val2  = rshm * rsfr / rshm
    gradVal = -C5 / val2

    spl,splgrad,br,r,mp,wm = stable_break(rs,medprof,[gradVal],idx=0,window_start=9)

    brs.append(br)

    currentMass += 0.1