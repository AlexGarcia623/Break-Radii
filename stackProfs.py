import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import h5py

#rString = 'rsfr10' 
#rString = 'rsfr50' 
rString = 'rshm'


hf  = h5py.File('Illustris_z0.h5','r') # File in -- all idv profiles
nF  = h5py.File('Illustris_z0med.h5','a') # File out -- stacked median profiles
nF2 = h5py.File('Illustris_z0%s.h5' %rString.upper(),'a') # File out -- Specific radius you want to create
# Create Stacked Median Profiles

for mbin in hf:
    print(mbin)
    hf1 = hf.get(mbin)

    all_prof_names = hf1.get('median_profiles')
    all_err_names  = hf1.get('error_bars')
    all_med_profs = []
    all_err_bars  = []
    for i in all_prof_names:
        all_med_profs.append(np.array(all_prof_names.get(i)))
    for i in all_err_names:
        all_err_bars.append(np.array(all_err_names.get(i)))
    
    all_med_profs = np.array(all_med_profs)
    all_err_bars  = np.array(all_err_bars)

    med_prof = np.nanmedian(all_med_profs,axis=0)

    err_bars = np.sqrt(np.nansum(1 / all_err_bars**2 * (all_med_profs - med_prof)**2,axis=0) 
                                            / np.nansum(1 / all_err_bars**2,axis=0))

    nF1 = nF.create_group(j)
    nF1.create_dataset('toterr',data = err_bars)
    nF1.create_dataset('totmed',data = med_prof)


profiles = [] 

for mbin in hf:
    profile = hf.get(mbin)
    rshm = []
    rs = profile.get(rString)
    for i in rs:
        rshm.append(np.array(rs.get(i)))
    median = np.median(np.array(rshm))
    g1 = h2.create_group(j)
    g1.create_dataset('median_%s' %rString,data=median)

    
hf.close()
nF.close()
nF2.close()
