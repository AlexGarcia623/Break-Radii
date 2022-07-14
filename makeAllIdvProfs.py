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

######## Example script for generating median profiles ########


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

def create_profs(run, base, snaps):
    '''Creates all individual median profiles
    
    IN:
    Run  -- Which Illustris/TNG run to use
    Base -- File directory where that run lives
    Snap -- Which snapshot
    
    OUT:
    h5 file with median profiles, errorbars and other useful information
    
    '''
    snap = snaps
    hdr  = il.groupcat.loadHeader(out_dir, snap)
    box_size = hdr['BoxSize']
    scf      = hdr['Time']
    print(scf)
    z        = (1.00E+00 / scf - 1.00E+00)
    print(z)
    
    fields = ['SubhaloGasMetallicity', 'SubhaloPos', 'SubhaloMass', 'SubhaloVel', 'SubhaloSFR',
              'SubhaloMassType','SubhaloGasMetallicitySfr','SubhaloHalfmassRadType']
    sub_cat = il.groupcat.loadSubhalos(out_dir, snap, fields = fields)
    grp_cat = il.groupcat.loadHalos(out_dir,snap,fields=['GroupFirstSub', 'Group_R_Crit200'])
    sub_cat['SubhaloMass'] *= 1.000E+10 / h
    sub_cat['SubhaloMassType'] *= 1.00E+10 / h
    
    subs     = np.array(grp_cat['GroupFirstSub'])
    
    subs = subs[(subs != 4294967295)]
    
    sfms_idx = sfmscut(sub_cat['SubhaloMassType'][subs,4], sub_cat['SubhaloSFR'][subs])
    
    subs     = subs[(sub_cat['SubhaloMassType'][subs,4] > 1.000E+01**m_star_min) & 
                 (sub_cat['SubhaloMassType'][subs,4] < 1.000E+01**m_star_max) &
                 (sub_cat['SubhaloMassType'][subs,0] > 1.000E+01**m_gas_min) &
                 (                                           sfms_idx) ]    
        
    mass = sub_cat['SubhaloMassType']
    
    stellarHalfmassRad = sub_cat['SubhaloHalfmassRadType'][:,4]
    
    stellarHalfmassRad *= (scf / h)
    
    bin0 = 8.5
    bin1 = bin0 + 0.5

    upperlimit = [40.0 ,65.0 ,100.0,150.0,200.0] # R_max 
#     upperlimit = [100.0,100.0,100.0,100.0,100.0] # R_max -- Illustris-1 z=0 is weird when stacked
    idx = 0 # index for upperlimit
    trackWhenSwitch = 0 # tracks when to change idx

    print(len(subs))
    endbin = 10.4
        
    hf = h5py.File('Illustris_z0.h5','a')
    
    while (bin0 < endbin):

        sub1 = subs[(mass[subs,4] > 1.00E+01**bin0) & (mass[subs,4] < 1.00E+01**bin1)]

    
        total_med_prof = []
        total_med_stds = []
        rs = []

        groupname = ''
        if str(bin0)[0] == '1':
            groupname = '1E+%s' %bin0
        else:
            groupname = '1E+0%s' %bin0

        g1   = hf.create_group(groupname)
        sg11 = g1.create_group('median_profiles')
        sg12 = g1.create_group('error_bars')
        sg13 = g1.create_group('rsfr50')
        sg14 = g1.create_group('rshm')
        sg15 = g1.create_group('rsfr10')

        bin0 += 0.1
        bin1 += 0.1
        elements = []
        for sub in sub1:
            pos,sfr = returnPosSfr(out_dir, snap, sub, sub_cat, box_size, scf)
            rsfr = calcrsfr(pos,sfr,frac = 5.000E-01)
            rin  = calcrsfr(pos,sfr,frac = 1.000E-01)
            sg13.create_dataset('sub_%s_rsfr50' %sub, data = rsfr)
            sg15.create_dataset('sub_%s_rin'    %sub, data = rin)
            sg14.create_dataset('sub_%s_rshm'   %sub, data = stellarHalfmassRad[sub])

            med_prof,med_stds = med_met(out_dir, snap, sub, sub_cat, box_size, scf, uplim=upperlimit[idx])
            sg11.create_dataset('sub_%s_med' %sub, data = med_prof)
            sg12.create_dataset('sub_%s_err' %sub, data = med_stds)
        trackWhenSwitch += 1
        if trackWhenSwitch == 5:
            idx += 1
            trackWhenSwitch = 0
            
    hf.close()
    
def med_met(out_dir, snap, sub, sub_cat, box_size, scf, uplim = 65.):
    '''Loads in particle data, converts to sensible units, and creates stacked profile
    
    '''
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

    gas_rad  = (gas_pos[:,0]**2 + gas_pos[:,1]**2 + gas_pos[:,2]**2)**(0.5) # For creating shells
    
    r  = 0    # kpc
    dr = 0.1  # kpc
    med_prof = []
    med_stds = []
    upper_lim = uplim 
    while r < upper_lim: 
        mask = ((gas_rad > r) & (gas_rad < r + dr))
    
        Z = OH[(mask)]         
        if len(Z) < 6: # If less than 6 particles in the shell -- np.NAN
            med_prof.append(np.NAN)
            med_stds.append(np.NAN)
        else:    
            med = np.nanmedian(Z)
            med_prof.append(np.log10(med)+12)
            met_std = np.sqrt( np.sum(np.array([i - med for i in Z])**2)/(len(Z)-1) )
          
            med_lstd = 1/2.303 * met_std/med
  
            med_stds.append(med_lstd)
        r += dr
        
    return med_prof, med_stds
    
def calc_sSFR(out_dir, snap, sub, sub_cat, box_size, scf):
    '''
    Global Specific Star Formation for a subhalo
    '''
    
    sfr       = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['StarFormationRate'])
    star_mass = il.snapshot.loadSubhalo(out_dir, snap, sub, 4, fields = ['Masses'           ])
    
    star_mass *= (1.000E+10 / h)
    
    
    return np.sum(sfr)/np.sum(star_mass)
    

def returnPosSfr(out_dir, snap, sub, sub_cat, box_size, scf):
    '''Gets the positions and SFRs for specific sub
    
    '''
    sub_pos = sub_cat['SubhaloPos'][sub]
    sub_met = sub_cat['SubhaloGasMetallicity'][sub]
    sub_vel = sub_cat['SubhaloVel'][sub]

    gas_pos   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Coordinates'      ])
    gas_sfr   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['StarFormationRate'])

    gas_pos    = center(gas_pos, sub_pos, box_size)
    gas_pos   *= (scf / h)

    return gas_pos,gas_sfr    

def line(data, p1, p2):
    '''Returns line
    
    y = p1 * data + p2
    y = m * x + b
    
    '''
    return p1*data + p2  

def trans(arr0, incl0):
    '''Orients everything to the face-on orientation
    
    '''
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
    '''Calculate inclination of galaxy
    
    '''
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
    '''Center the galaxy on (0, 0, 0)
    
    '''
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
    '''Calculate rsfr in and out (see Hemler et al., 2021)
    
    '''
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
    '''Create star-formation main sequence, select only star-forming galaxies
    
    '''
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
#     mbrk    = 1.0200E+01
#     mstp    = 2.0000E-01
    mbrk    = m_star_max
    mstp    = 1.000E-01
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
    mmax    = m_star_max
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
    '''Calculate radius enclosing certain fraction of SFR
    
    '''
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

# Run code 

for snap1 in snapshots:
    create_profs(run, base, snap1)
