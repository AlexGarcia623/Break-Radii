# Break-Radii


### Most of the code associated Garcia et al. (2022)


Data in the h5 files are associated with Figures 5 (TNG50), 8 (TNG50-2), and 10 (Illustris)
   - All data points are in physical kpc and are sorted in increasing mass order (i.e. 8.5, 8.6 ..., 10.4) and in increasing redshift order (i.e. z=0, 0.5, 1, 2, and 3).
   - The stellar half mass radii are also included in these files. They are also in physical kpc
   - To regenerate these
       - all individual profiles will need to be recreated using `makeAllIdvProfs.py`
       - profiles (and rsfr/rshm) will need to be stacked with `stackProfs.py`
       - fit the profiles with `fitting.py`