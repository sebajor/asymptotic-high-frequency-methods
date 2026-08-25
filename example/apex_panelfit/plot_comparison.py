import sys
sys.path.append('../../source')
from hyperparameters import *
from plot_utils import plot_deformations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import h5py
#from plot_utils import plot_deformations
from hyperparameters import *

fit_filename = 'images_20260502_20365000/images_geo_aperture_conj/images_panels_10/params.npz'
classical_filename = 'images_20260502_20365000/Data/data.hdf5'

subref_filename =  'images_20260502_20365000/images_geo_aperture_conj/images_panels_sec_adjust/params.npz'


##first we will create the apex geometry
panels, s_pos, s_n, s_ds, sec_vertex, B, target_pos = create_apex_geometries(r_min_prim, d1, r_points,
                      r_min_sec, d2, 
                      t_points_primary, t_points_secondary,
                      f1, f_d, 
                      silhouette, legs_diameter, secondary_silhouette,
                      sigma_t, sigma_r,
                      target_distance, target_map_size, target_points,
                      batch_size=batch_size
                      )


##I will just read the
f = h5py.File(classical_filename, 'r')

x = np.array(f['grids']['x'])
y = np.array(f['grids']['y'])

r = np.sqrt(x**2+y**2)
mask = np.bitwise_and((r<6), r>0.375)

mask_plot = np.ones(mask.shape)*np.nan
mask_plot[mask] = 1

surf5 = np.array(f['legs_removal']['corrected'])

surf6 =np.array(f['panel_fitting_deform']['model'])
p_mask = np.array(f['panel_fitting_deform']['panel_mask'])
p_mask_plot = np.ones(p_mask.shape)*np.nan
p_mask_plot[p_mask.astype(bool)] = 1
f.close()

###get the coeffs
f = np.load(fit_filename, allow_pickle=1)
f_subref = np.load(subref_filename, allow_pickle=1)



##these are the plots of the panels

coeffs = f['params'].tolist()['coeffs']
coeffs_subref = f_subref['params'].tolist()['coeffs']
##we will swap the x,y signs in the panels to have the good old plot
for p in panels.keys():
    p0 = panels[p]['p0']
    p0[:,0] *= -1
    p0[:,1] *= -1
    panels[p]['p0'] = p0

fig, axes = plt.subplots(1,2, sharex=1, sharey=1)
axes[1].pcolormesh(x,y, surf5*mask_plot, vmin=-50, vmax=50, cmap=cm.jet)
plot_deformations(panels, coeffs, ax=axes[0], correct_global=False, pol_deg=2)

axes[0].set_aspect('equal')
axes[1].set_aspect('equal')
axes[0].set_title('Kirchhoff-Fresnel optimization')
axes[1].set_title('Classical Fourier method')

#fig, axes = plt.subplots(2,2)
#plot_deformations(panels, coeffs, ax=axes[0,0], correct_global=False, pol_deg=2)
#plot_deformations(panels, coeffs, ax=axes[0,1], correct_global=True, pol_deg=2)
#axes[1,0].pcolormesh(x,y, surf5*mask_plot, vmin=-50, vmax=50, cmap=cm.jet)
#axes[1,1].pcolormesh(x,y, surf6*p_mask_plot, vmin=-50, vmax=50, cmap=cm.jet)



plt.figure()
plt.plot(f['loss'])
plt.grid()
plt.title('Loss evolution')
plt.ylabel("Loss")
plt.xlabel("iterations")


fig, axes = plt.subplots(1,2, sharex=1, sharey=1)
plot_deformations(panels, coeffs, ax=axes[0], correct_global=False, pol_deg=2)
plot_deformations(panels, coeffs_subref, ax=axes[1], correct_global=False, pol_deg=2)

axes[0].set_aspect('equal')
axes[1].set_aspect('equal')
axes[0].set_title('Fixed geometry')
axes[1].set_title('Subreflector parameters free')


plt.figure()
plt.plot(f_subref['loss'])
plt.grid()
plt.title('Loss evolution')
plt.ylabel("Loss")
plt.xlabel("iterations")

