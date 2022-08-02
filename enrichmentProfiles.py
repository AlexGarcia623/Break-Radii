import sys
import os
import time
import h5py
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import illustris_python as il
from scipy.optimize import curve_fit
import scipy.stats
from scipy.stats import linregress
from scipy import interpolate
from scipy.interpolate import UnivariateSpline

######## Example script for generating stacked enrichment profiles ########

######## Which Illustris to use?

# run = 'L35n2160TNG' #TNG50 - 1
run = 'L35n1080TNG' #TNG50 - 2
# run = 'L75n1820FP' # Illustris - 1
# run = 'L75n1820TNG' #TNG100 - 1

######## File directory for Data on HPG

# base = '/orange/paul.torrey/Illustris/Runs/' + run # Illustris -- Paul's directory
# out_dir = base

# base = '/orange/paul.torrey/zhemler/IllustrisTNG/' + run + '/' # Zach's directory -- TNG50 z=0,1,2,3
# out_dir = base + 'output/' # needed for Zach's directory

base = '/orange/paul.torrey/IllustrisTNG/Runs/' + run  # Paul's directory -- TNG50 z=0.5
out_dir = base

# When using Paul's directory need to edit Illustris Python as well


######## Which snapshot(s) 

snapshots = [99]  # z=0 TNG
# snapshots = [135] # z=0 Illustris

######## Constants

h      = 6.774E-01
xh     = 7.600E-01
zo     = 3.500E-01
mh     = 1.6726219E-24
kb     = 1.3806485E-16
mc     = 1.270E-02

m_star_min = 8.5
m_star_max = 10.9
m_gas_min  = 8.5

def create_profs(run, base, out_dir, snap):
    snap = snaps[0]
    hdr  = il.groupcat.loadHeader(out_dir, snap)
    box_size = hdr['BoxSize']
    scf      = hdr['Time']
    print(scf)
    z0       = (1.00E+00 / scf - 1.00E+00)
    print(z0)
    fields = ['SubhaloGasMetallicity', 'SubhaloPos', 'SubhaloMass', 'SubhaloVel', 'SubhaloSFR',
              'SubhaloMassType','SubhaloGasMetallicitySfr','SubhaloHalfmassRadType']
    sub_cat = il.groupcat.loadSubhalos(out_dir, snap, fields = fields)
    grp_cat = il.groupcat.loadHalos(out_dir,snap,fields=['GroupFirstSub', 'Group_R_Crit200'])
    sub_cat['SubhaloMass'] *= 1.000E+10 / h
    sub_cat['SubhaloMassType'] *= 1.00E+10 / h

    subs     = np.array(grp_cat['GroupFirstSub'])
    
    subs = subs[(subs != 4294967295)] # This resolves an issue within Illustris

    sfms_idx = sfmscut(sub_cat['SubhaloMassType'][subs,4], sub_cat['SubhaloSFR'][subs])
    subs     = subs[(sub_cat['SubhaloMassType'][subs,4] > 1.000E+01**m_star_min) & 
                 (sub_cat['SubhaloMassType'][subs,4] < 1.000E+01**m_star_max) &
                 (sub_cat['SubhaloMassType'][subs,0] > 1.000E+01**m_gas_min) &
                 (                                           sfms_idx) ]    

    
    mass = sub_cat['SubhaloMassType']

    bin0 = 8.5
    bin1 = bin0 + 5.000E-01

    upperlimit = [40.0,65.0,100.0,150.0,200.0]
    idx = 0
    j   = 0

    endbin = 10.4
    
    hf       = h5py.File('TNG50Enrich_z0_newAlpha.h5','a')
    RSHMfile = h5py.File('../Illustris/TNG50_z0RSHM.h5','r')
    RSFRfile = h5py.File('../Illustris/TNG50_z0RSFR.h5','r')
    
    while (bin0 < endbin):
        sub1 = subs[(mass[subs,4] > 1.00E+01**bin0) & (mass[subs,4] < 1.00E+01**bin1)]
        groupname = ''
        if str(bin0)[0] == '1':
            groupname = '1E+%s' %bin0
        else:
            groupname = '1E+0%s' %bin0
        g1   = hf.create_group(groupname)
        bin0 += 1.000E-01
        bin1 += 1.000E-01
  
        # Create lists for stacking

        Zs      = [] # log O/H + 12
        dZdts   = [] # Rate of change of Z -- see Garcia+22
        Mgass   = [] # Gas Mass
        SFRs    = [] # SFR
        profs   = [] # Idv median profiles
        rins    = [] # Rin -- encloses 10% of SFR
        totdens = [] # Density
        vr      = [] # Radial Velocity
        vels    = [] # Velocity
        srs     = [] # Radial velocity dispersion
        sigmas  = [] # Velocity Dispersion
        
        # Get individual enrichment profiles
        
        for sub in sub1:
            Z,Mgas,SFR,Dens,vrs,sr,vs,sigma = enrich_profs(out_dir, snap, sub, sub_cat, box_size, scf, uplim=upperlimit[idx])
            Zs.append(Z)
            Mgass.append(Mgas)
            SFRs.append(SFR)
            totdens.append(Dens)
            srs.append(sr)
            sigmas.append(sigma)
            vr.append(vrs)
            vels.append(vs)
            
            pos,sfr = returnPosSfr(out_dir, snap, sub, sub_cat, box_size, scf)
            rin = calcrsfr(pos,sfr,frac = 1.000E-01)
            
            rins.append(rin)
            
            # Profile stuff - I only care about the median for this for now
            mprof = met_profs(out_dir, snap, sub, sub_cat, box_size, scf, uplim=upperlimit[idx])
            profs.append(mprof)
        
        # Stack the profiles,find the break radius
        
        profs = np.array(profs)
        stack_mprof = np.nanmedian(profs,axis=0)
        rin = np.nanmedian(rins)
        
        rs = np.arange(0,len(stack_mprof)/10.0,0.1)
        
        stack_mprof = stack_mprof[(rs > rin)]
        rs          =          rs[(rs > rin)]
        
        C = 0.28
        C /= 3.5
        
        rsfr = float(np.array(RSFRfile.get(groupname).get('median_rsfr50')))
        rshm = float(np.array(RSHMfile.get(groupname).get('median_rshm')))
        
        gradVal = -C / ( rshm ) * ( rshm / rsfr )
        
        spl2Use,splgrad2Use,br2Use,rs2Use,mp2Use,window_mid = stable_break(rs,stack_mprof,[gradVal],window_start=9)
        
        # Stack the profiles, save them
        
        Zs     = np.array(Zs)
        Mgass  = np.array(Mgass)
        SFRs   = np.array(SFRs)
        dens   = np.array(totdens)
        vrS    = np.array(vr)
        Vels   = np.array(vels)
        srs    = np.array(srs)
        sigmas = np.array(sigmas)

        Z      = np.nanmedian(Zs    ,axis=0)
        dZdt   = np.nanmedian(dZdts ,axis=0)
        Mgas   = np.nanmedian(Mgass ,axis=0)
        vr     = np.nanmedian(vrS   ,axis=0)
        vels   = np.nanmedian(Vels  ,axis=0)
        srs    = np.nanmedian(srs)
        sigmas = np.nanmedian(sigmas)
        
        SFRs  = np.nanmean(SFRs,axis=0)
        Dens  = np.nanmean(Dens,axis=0)
        
        encMass = find_Menc(Dens,rs,br2Use)
        
        dZdt  = 1/Mgas * 0.05 * SFRs

        tau   = Z/dZdt
        
        g1.create_dataset('Enrichment Timescale Profile',data=tau)
        g1.create_dataset('Dynamical Timescale', data = Tcross(encMass,br2Use))
        g1.create_dataset('Break Radius (kpc)',data=br2Use)
        g1.create_dataset('Gas Velocity (km/s)',data=vels)
        g1.create_dataset('Radial Velocity (km/s)',data=vr)
        g1.create_dataset('Radial Velocity Dispersion',data=srs)
        g1.create_dataset('Velocity Dispersion',data=sigmas)
        
        j += 1
        if j == 5:
            idx += 1
            j = 0    

    hf.close()
    RSHMfile.close()
    RSFRfile.close()

def enrich_profs(out_dir, snap, sub, sub_cat, box_size, scf, uplim = 65.):

    sub_pos = sub_cat['SubhaloPos'][sub]
    sub_met = sub_cat['SubhaloGasMetallicity'][sub]
    sub_vel = sub_cat['SubhaloVel'][sub]

    gas_pos   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Coordinates'      ])
    gas_vel   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Velocities'       ])
    gas_mass  = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Masses'           ])
    gas_sfr   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['StarFormationRate'])
    gas_rho   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Density'          ])
    gas_met   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metallicity'  ])
    star_mass = il.snapshot.loadSubhalo(out_dir, snap, sub, 4, fields = ['Masses'           ])
    tot_dens  = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['SubfindDensity'   ])
    ZO        = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])[:,4]
    XH        = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])[:,0]

    gas_pos    = center(gas_pos, sub_pos, box_size)
    gas_pos   *= (scf / h)
    gas_vel   *= np.sqrt(scf)
    gas_vel   -= sub_vel
    gas_mass  *= (1.000E+10 / h)
    star_mass *= (1.000E+10 / h)
    gas_rho   *= (1.000E+10 / h) / (scf / h )**3.00E+00
    gas_rho   *= (1.989E+33    ) / (3.086E+21**3.00E+00) # cgs units
    tot_dens  *= (1.000E+10 / h) / (scf / h )**3.00E+00
    tot_dens  *= (1.989E+33    ) / (3.086E+21**3.00E+00)
    gas_rho   *= xh / mh
    
    OH = ZO/XH * 1.00/16.00

    ri, ro = calc_rsfr_io(gas_pos, gas_sfr)
    ro2    = 2.000E+00 * ro

    sf_idx = gas_rho > 1.300E-01
    incl   = calc_incl(gas_pos[sf_idx], gas_vel[sf_idx], gas_mass[sf_idx], ri, ro)

    gas_pos  = trans(gas_pos, incl)
    gas_vel  = trans(gas_vel, incl)

    gas_rad  = np.sqrt(gas_pos[:,0]**2 + gas_pos[:,1]**2 + gas_pos[:,2]**2)
    
    radvel   = (gas_pos[:,0] * gas_vel[:,0] + gas_pos[:,1] * gas_vel[:,1]) / np.sqrt(gas_pos[:,0]**2 + gas_pos[:,1]**2)
    sigma_r  = np.std(radvel)
    
    gas_vtot = np.sqrt(gas_vel[:,0]**2.0 + gas_vel[:,1]**2 + gas_vel[:,2]**2)
    sigma    = np.std(np.sqrt(gas_vel[:,0]**2.0 + gas_vel[:,1]**2 + gas_vel[:,2]**2))

    r  = 0
    dr = 1.000E-01
    y  = 0.05 # Cited from Paul's paper: Torrey et al (2018)
    Mgas    = []
    SFRs    = []
    Zs      = []
    dens    = []
    vrs     = []
    vtots   = []   
    while r <= uplim:
        mask  = ((gas_rad > r) & (gas_rad < r + dr))
        Z     =        OH[(mask)] ## OH versus gas_met -- gas-phase metallicity##
        gasM  =  gas_mass[(mask)]
        SFR   =   gas_sfr[(mask)]
        den   =  tot_dens[(mask)]
        vr    =    radvel[(mask)]
        vtot  =  gas_vtot[(mask)]
        
        if len(Z) < 6:
            Zs.append(np.nan)
            Mgas.append(np.nan)
            SFRs.append(np.nan)
            dens.append(np.nan)
            vrs.append(np.nan)
            vtots.append(np.nan)
        else:
            Zs.append(np.nanmedian(np.array(Z)))
            Mgas.append(np.nanmedian(np.array(gasM)))
            SFRs.append(np.nanmean(np.array(SFR))) # Mean not median
            dens.append(np.nanmean(np.array(den))) # Mean not median
            
            vrs.append(np.nanmedian(np.array(vr)))
            vtots.append(np.nanmedian(np.array(vtot)))
        r+=dr

    return np.array(Zs),np.array(Mgas),np.array(SFRs),np.array(dens),np.array(vrs),sigma_r,np.array(vtots),sigma


def find_Menc(dens,rs,r):
    '''Return enclosed mass in cgs units
    '''
    rs *= 3.086E+21
    r  *= 3.086E+21
    Menc = 4/3 * np.pi * rs**3.00E+00 * dens    
    return Menc[(rs < r)][-1]

def Vcirc(mass,radius):
    '''Return circular velocity in cgs units
    '''
    G = 6.67e-8
    radius *= 3.086e+21
    return np.sqrt(G*mass/radius)
                         
def Tcirc(mass,radius):
    '''Return circular time in cgs units
    '''
    vcirc = Vcirc(mass,radius)
    radius *= 3.086e+21
    return 2.0*np.pi*radius / vcirc
                          
def Tcross(mass,radius):
    '''Return crossing time in cgs units
    '''
    vcirc = Vcirc(mass,radius)
    radius *= 3.086e+21
    return radius/vcirc

############### Fitting routine, check documentation in fitting.py for specifics

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
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

def find_closest(rs,grad,value,margin=2e-3):
    lookAt = rs[(np.abs(grad - value) < margin) ]
    if len(lookAt) == 0:
        return np.nan
    breakidx = [int(np.where(rs == lookAt[i])[0]) for i in range(len(lookAt))]
    for idx in breakidx:
        if (grad[idx - 1] < grad[idx]):
            return rs[idx]
    return np.nan


def adapted_spline(rs,mprof,window=55,grad_val=-0.02):
    mprof = np.array(mprof)
        
    rs    =    rs[(~np.isnan(mprof))] 
    mprof = mprof[(~np.isnan(mprof))]
    
    mprof = mprof[np.argsort(rs)]
    rs    =    rs[np.argsort(rs)]
    
    ### Profiles
    
    smoothed = savitzky_golay(mprof, window_size=window, order = 7) # order 7 is arbitrary pick    
    spl = UnivariateSpline(rs,smoothed)    
    spl.set_smoothing_factor(0.1)


    ### Gradient 
    splgrad = spl.derivative(1)
    
    ### Break Radius
    break_val = find_closest(rs,splgrad(rs),grad_val)
            
    return spl(rs), splgrad(rs), break_val, rs, mprof

                        
def stable_break(rsfrs,mprofs,grads,window_start=9:
    keepGoing = True
    window_mid = window_start + 4
    spl2Use = None
    splgrad2use = None
    br2Use = None
    rs2Use = None
    mp2Use = None
    while keepGoing:
        spl1,splgrad1,br1,rs1,mprof1 = adapted_spline(rsfrs,mprofs,grad_val=grads,window=window_mid-4)
        spl2,splgrad2,br2,rs2,mprof2 = adapted_spline(rsfrs,mprofs,grad_val=grads,window=window_mid-2)
        spl3,splgrad3,br3,rs3,mprof3 = adapted_spline(rsfrs,mprofs,grad_val=grads,window=window_mid)
        spl4,splgrad4,br4,rs4,mprof4 = adapted_spline(rsfrs,mprofs,grad_val=grads,window=window_mid+2)
        spl5,splgrad5,br5,rs5,mprof5 = adapted_spline(rsfrs,mprofs,grad_val=grads,window=window_mid+4)
                
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
            
        if window_mid > 100: # Trouble Shooting
            keepGoing = False
            window_mid = np.nan
            spl2Use = np.nan
            splgrad2Use = np.nan
            br2Use = np.nan
            rs2Use = np.nan
            mp2Use = np.nan
            print('Over 10.0 kpc, returning NaN')
    return spl2Use,splgrad2Use,br2Use,rs2Use,mp2Use,window_mid

############### Everything below here is the same as it is in makeAllIdvProfs.py, check documentation there

def returnPosSfr(out_dir, snap, sub, sub_cat, box_size, scf):
    sub_pos = sub_cat['SubhaloPos'][sub]
    sub_met = sub_cat['SubhaloGasMetallicity'][sub]
    sub_vel = sub_cat['SubhaloVel'][sub]

    gas_pos   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Coordinates'      ])
    gas_sfr   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['StarFormationRate'])

    gas_pos    = center(gas_pos, sub_pos, box_size)
    gas_pos   *= (scf / h)

    return gas_pos,gas_sfr
                         
def met_profs(out_dir, snap, sub, sub_cat, box_size, scf, uplim = 65.):
    sub_pos = sub_cat['SubhaloPos'][sub]
    sub_met = sub_cat['SubhaloGasMetallicity'][sub]
    sub_vel = sub_cat['SubhaloVel'][sub]

    gas_pos   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Coordinates'      ])
    gas_vel   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Velocities'       ])
    gas_mass  = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Masses'           ])
    gas_sfr   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['StarFormationRate'])
    gas_rho   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Density'          ])
    gas_met   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metallicity'  ])
    ZO        = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])[:,4]
    XH        = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])[:,0]
    
    gas_pos    = center(gas_pos, sub_pos, box_size)
    gas_pos   *= (scf / h)
    gas_vel   *= np.sqrt(scf)
    gas_vel   -= sub_vel
    gas_mass  *= (1.000E+10 / h)
    gas_rho   *= (1.000E+10 / h) / (scf / h )**3.00E+00
    gas_rho   *= (1.989E+33    ) / (3.086E+21**3.00E+00)
    gas_rho   *= xh / mh

    OH = ZO/XH * 1.00/16.00

    ri, ro = calc_rsfr_io(gas_pos, gas_sfr)
    ro2    = 2.000E+00 * ro

    sf_idx = gas_rho > 1.300E-01
    incl   = calc_incl(gas_pos[sf_idx], gas_vel[sf_idx], gas_mass[sf_idx], ri, ro)

    gas_pos  = trans(gas_pos, incl)
    gas_vel  = trans(gas_vel, incl)

    gas_rad  = (gas_pos[:,0]**2 + gas_pos[:,1]**2 + gas_pos[:,2]**2)**(0.5)

    r  = 0    # kpc
    dr = 0.1  # kpc
    med_prof = []
    upper_lim = uplim 
    while r < upper_lim: 
        mask = ((gas_rad > r) & (gas_rad < r + dr))
        Z = OH[(mask)]         
        if len(Z) < 6:
            med_prof.append(np.NAN)
        else:    
            med = np.nanmedian(Z)
            med_prof.append(np.log10(med)+12)
        r += dr
    return np.array(med_prof)
                         
def line(data, p1, p2):
    return p1*data + p2      
                         
def trans(arr0, incl0):
    arr      = np.copy( arr0)
    incl     = np.copy(incl0)
    deg2rad  = np.pi / 1.800E+02
    incl    *= deg2rad
    arr[:,0] = -arr0[:,2] * np.sin(incl[0]) + (arr0[:,0] * np.cos(incl[1]) + arr0[:,1] * np.sin(incl[1])) * np.cos(incl[0])
    arr[:,1] = -arr0[:,0] * np.sin(incl[1]) + (arr0[:,1] * np.cos(incl[1])                                                )
    arr[:,2] =  arr0[:,2] * np.cos(incl[0]) + (arr0[:,0] * np.cos(incl[1]) + arr0[:,1] * np.sin(incl[1])) * np.sin(incl[0])
    del incl
    return arr

def calc_incl(pos0, vel0, m0, ri, ro):
    rpos = np.sqrt(pos0[:,0]**2.000E+00 +
                   pos0[:,1]**2.000E+00 +
                   pos0[:,2]**2.000E+00 )
    idx  = (rpos > ri) & (rpos < ro)
    pos  = pos0[idx]
    vel  = vel0[idx]
    m    =   m0[idx]

    hl = np.cross(pos, vel)
    L  = np.array([np.multiply(m, hl[:,0]),
                   np.multiply(m, hl[:,1]),
                   np.multiply(m, hl[:,2])])
    L  = np.transpose(L)
    L  = np.array([np.sum(L[:,0]),
                   np.sum(L[:,1]),
                   np.sum(L[:,2])])
    Lmag  = np.sqrt(L[0]**2.000E+00 +
                    L[1]**2.000E+00 +
                    L[2]**2.000E+00 )
    Lhat  = L / Lmag
    incl  = np.array([np.arccos(Lhat[2]), np.arctan2(Lhat[1], Lhat[0])])
    incl *= 1.800E+02 / np.pi
    if   incl[1]  < 0.000E+00:
         incl[1] += 3.600E+02
    elif incl[1]  > 3.600E+02:
         incl[1] -= 3.600E+02
    return incl

                         
def center(pos0, centpos, boxsize = None):
    pos       = np.copy(pos0)
    pos[:,0] -= centpos[0]
    pos[:,1] -= centpos[1]
    pos[:,2] -= centpos[2]
    if (boxsize != None):
        pos[:,0][pos[:,0] < (-boxsize / 2.000E+00)] += boxsize
        pos[:,0][pos[:,0] > ( boxsize / 2.000E+00)] -= boxsize
        pos[:,1][pos[:,1] < (-boxsize / 2.000E+00)] += boxsize
        pos[:,1][pos[:,1] > ( boxsize / 2.000E+00)] -= boxsize
        pos[:,2][pos[:,2] < (-boxsize / 2.000E+00)] += boxsize
        pos[:,2][pos[:,2] > ( boxsize / 2.000E+00)] -= boxsize
    return pos

def calc_rsfr_io(pos0, sfr0):
    fraci = 5.000E-02
    fraco = 9.000E-01
    r0    = 1.000E+01
    rpos  = np.sqrt(pos0[:,0]**2.000E+00 +
                    pos0[:,1]**2.000E+00 +
                    pos0[:,2]**2.000E+00 )
    sfr   = sfr0[np.argsort(rpos)]
    rpos  = rpos[np.argsort(rpos)]
    sfrtot = np.sum(sfr)
    if (sfrtot < 1.000E-09):
        return np.nan, np.nan
    sfrf   = np.cumsum(sfr)/sfrtot
    idx0   = np.arange(1, len(sfr) + 1, 1)
    idxi   = idx0[(sfrf > fraci)]
    idxi   = idxi[0]
    rsfri  = rpos[idxi]
    dskidx = rpos < (rsfri + r0)
    sfr    =  sfr[dskidx]
    rpos   = rpos[dskidx]
    sfrtot = np.sum(sfr)
    if (sfrtot < 1.000E-09):
        return np.nan, np.nan
    sfrf   = np.cumsum(sfr) / sfrtot
    idx0   = np.arange(1, len(sfr) + 1, 1)
    idxo   = idx0[(sfrf > fraco)]
    idxo   = idxo[0]
    rsfro  = rpos[idxo]
    return rsfri, rsfro

def sfmscut(m0, sfr0):
    nsubs = len(m0)
    idx0  = np.arange(0, nsubs)
    non0  = ((m0   > 0.000E+00) & 
             (sfr0 > 0.000E+00) )
    m     =    m0[non0]
    sfr   =  sfr0[non0]
    idx0  =  idx0[non0]
    ssfr  = np.log10(sfr/m)
    sfr   = np.log10(sfr)
    m     = np.log10(  m)

    idxbs   = np.ones(len(m), dtype = int) * -1
    cnt     = 0
    mbrk    = 1.020E+01
    mstp    = 2.000E-01
    mmin    = m_star_min
    mbins   = np.arange(mmin, mbrk + mstp, mstp)
    rdgs    = []
    rdgstds = []


    for i in range(0, len(mbins) - 1):
        idx   = (m > mbins[i]) & (m < mbins[i+1])
        idx0b = idx0[idx]
        mb    =    m[idx]
        ssfrb = ssfr[idx]
        sfrb  =  sfr[idx]
        rdg   = np.median(ssfrb)
        idxb  = (ssfrb - rdg) > -5.000E-01
        lenb  = np.sum(idxb)
        idxbs[cnt:(cnt+lenb)] = idx0b[idxb]
        cnt += lenb
        rdgs.append(rdg)
        rdgstds.append(np.std(ssfrb))

    rdgs       = np.array(rdgs)
    rdgstds    = np.array(rdgstds)
    mcs        = mbins[:-1] + mstp / 2.000E+00

    parms, cov = curve_fit(line, mcs, rdgs, sigma = rdgstds)
    mmin    = mbrk
    mmax    = 1.100E+01
    mbins   = np.arange(mmin, mmax + mstp, mstp)
    mcs     = mbins[:-1] + mstp / 2.000E+00
    ssfrlin = line(mcs, parms[0], parms[1])
    for i in range(0, len(mbins) - 1):
        idx   = (m > mbins[i]) & (m < mbins[i+1])
        idx0b = idx0[idx]
        mb    =    m[idx]
        ssfrb = ssfr[idx]
        sfrb  =  sfr[idx]
        idxb  = (ssfrb - ssfrlin[i]) > -5.000E-01
        lenb  = np.sum(idxb)
        idxbs[cnt:(cnt+lenb)] = idx0b[idxb]
        cnt += lenb
    idxbs    = idxbs[idxbs > 0]
    sfmsbool = np.zeros(len(m0), dtype = int)
    sfmsbool[idxbs] = 1
    sfmsbool = (sfmsbool == 1)
    return sfmsbool

def calcrsfr(pos0, sfr0, frac = 5.000E-01, ndim = 3):
    if (ndim == 2):
        rpos = np.sqrt(pos0[:,0]**2.000E+00 + 
                       pos0[:,1]**2.000E+00 )
    if (ndim == 3):
        rpos = np.sqrt(pos0[:,0]**2.000E+00 + 
                       pos0[:,1]**2.000E+00 +
                       pos0[:,2]**2.000E+00 )    
    sfr    = sfr0[np.argsort(rpos)]
    rpos   = rpos[np.argsort(rpos)]
    sfrtot = np.sum(sfr)
    if (sfrtot < 1.000E-09):
        return np.nan
    sfrf   = np.cumsum(sfr) / sfrtot
    idx0   = np.arange(1, len(sfr) + 1, 1)
    idx    = idx0[(sfrf > frac)]
    idx    = idx[0]
    rsfr50 = rpos[idx]
    return rsfr50

create_profs(run, base, out_dir, snaps)
