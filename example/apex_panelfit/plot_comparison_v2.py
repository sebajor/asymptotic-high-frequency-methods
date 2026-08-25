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

proc_filename = 'images_20260502_20365000_panel_beam/Data/data.hdf5'


subref_filename =  'images_20260502_20365000/images_geo_aperture_conj/images_panels_sec_adjust/params.npz'



def APEX_get_panel_nodes(
    rings = [0.375, 1.265, 1.820, 2.605, 3.220, 4.040, 4.780, 5.435, 6.000],
    ring_name = [  'A',   'B',   'C',   'D',   'E',   'F',   'G',   'H'  ],
    n_panels = [   12,    12,    24,    24,    48,    48,    48,    48  ]
        ):
    """
    Calculate the initial/ending radius/angle for each panel
    """
    r_init = []
    r_end = []
    angle_init = []
    angle_end = []
    v_number = []
    v_id = []
    names = []

    for i in range(8):
        panels = n_panels[i]
        theta = np.linspace(0, 2*np.pi, panels+1)
        r0 = rings[i]*np.ones(panels)
        theta0 = theta[:-1]
        r1 = rings[i+1]*np.ones(panels)
        theta1 = theta[1:]
        ##
        vnumber = np.flipud( np.roll(np.arange(panels)+1 ,-panels//4))  ##This is bcs the panel numbering is the worst
        vid = 100*(i+1)+vnumber
        name = [ring_name[i]+str(x) for x in range(1,panels+1)]
        r_init += r0.tolist()
        r_end += r1.tolist()
        angle_init += theta0.tolist()
        angle_end += theta1.tolist()
        v_number += vnumber.tolist()
        v_id += vid.tolist()
        names += name

    return names, r_init, r_end, angle_init, angle_end, v_number, v_id


def APEX_draw_panels(ax, alpha=0.5, show_names=False,
        rings = [0.375, 1.265, 1.820, 2.605, 3.220, 4.040, 4.780, 5.435, 6.000],
        ring_name = [  'A',   'B',   'C',   'D',   'E',   'F',   'G',   'H'  ],
        n_panels = [   12,    12,    24,    24,    48,    48,    48,    48  ]
        ):
    names, r_i, r_e, a_i, a_e, vn, vi =  APEX_get_panel_nodes(rings=rings, ring_name=ring_name,
                                                         n_panels=n_panels)

    ##first we need to draw the lines in r
    x_r = [r_i*np.cos(a_i), r_e*np.cos(a_i)]
    y_r = [r_i*np.sin(a_i), r_e*np.sin(a_i)]

    for x,y in zip(x_r,y_r):
        ax.plot(x_r, y_r, alpha=alpha, lw=0.5, c=[0.6,0.6,0.6])

    if(show_names):
        xlabel = (np.array(r_e)-0.1)*np.cos((np.array(a_i)+np.array(a_e))/2)
        ylabel = (np.array(r_e)-0.1)*np.sin((np.array(a_i)+np.array(a_e))/2)
        rotation = np.arctan2(ylabel, xlabel)*180/np.pi-90
        for x,y,panel_id, rot in zip(xlabel, ylabel, vi, rotation):
            ax.text(x,y,panel_id, fontsize=6, color=(0.8,0.8,0.8,1),
                    ha='center',va='center', rotation=rot)
    ##now we need to join the radius
    radii = np.unique(np.append(r_i, rings[-1]))
    theta = np.linspace(-np.pi, np.pi, 361)
    for r in radii:
        ax.plot(r*np.cos(theta), r*np.sin(theta), alpha=alpha, lw=0.5,
                color=(0.6,0.6,0.6))
    ##support legs
    ct = np.cos(theta)
    st = np.sin(theta)
    cc = (0.6,0.6,0.6)

    ax.fill( 0.1*ct-4.6,0.05*st,color=cc,lw=0.5)
    ax.fill( 0.1*ct-4.3,0.05*st,color=cc,lw=0.5)
    ax.fill( 0.1*ct-3.0,0.05*st,color=cc,lw=0.5)
    ax.fill( 0.1*ct-5.8,0.05*st,color=cc,lw=0.5)

    ax.fill( 0.1*ct+4.6,0.05*st,color=cc,lw=0.5)
    ax.fill( 0.1*ct+4.3,0.05*st,color=cc,lw=0.5)
    ax.fill( 0.1*ct+3.0,0.05*st,color=cc,lw=0.5)
    ax.fill( 0.1*ct+5.8,0.05*st,color=cc,lw=0.5)

    ax.fill( 0.05*ct,0.1*st+4.6,color=cc,lw=0.5)
    ax.fill( 0.05*ct,0.1*st+4.3,color=cc,lw=0.5)
    ax.fill( 0.05*ct,0.1*st+3.0,color=cc,lw=0.5)
    ax.fill( 0.05*ct,0.1*st+5.8,color=cc,lw=0.5)

    ax.fill( 0.05*ct,0.1*st-4.6,color=cc,lw=0.5)
    ax.fill( 0.05*ct,0.1*st-4.3,color=cc,lw=0.5)
    ax.fill( 0.05*ct,0.1*st-3.0,color=cc,lw=0.5)
    ax.fill( 0.05*ct,0.1*st-5.8,color=cc,lw=0.5)



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

mask = r<6
mask_plot2 = np.ones(mask.shape)*np.nan
mask_plot2[mask] = 1


surf5 = np.array(f['legs_removal']['corrected'])

surf6 =np.array(f['panel_fitting_deform']['model'])
p_mask = np.array(f['panel_fitting_deform']['panel_mask'])
p_mask_plot = np.ones(p_mask.shape)*np.nan
p_mask_plot[p_mask.astype(bool)] = 1
f.close()
##fitted data process by the fourier pipeline
f = h5py.File(proc_filename, 'r')
surf5_cook = np.array(f['legs_removal']['corrected'])
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


fig, axes = plt.subplots(1,3, sharex=1, sharey=1)
im = axes[0].pcolormesh(x,y, surf5*mask_plot, vmin=-50, vmax=50, cmap=cm.jet)
plot_deformations(panels, coeffs, ax=axes[1], correct_global=True, pol_deg=4)
axes[2].pcolormesh(x,y, surf5_cook*mask_plot, vmin=-50, vmax=50, cmap=cm.jet)

for ax in axes.flatten():
    ax.set_aspect('equal')
    APEX_draw_panels(ax)

axes[1].set_title('Kirchhoff-Fresnel optimization\n RMS:27.64um')
axes[0].set_title('Classical Fourier method over measurement\n RMS:29.19um')
axes[2].set_title('Fourier method over optimized beam via Kirchhoff-Fresnel\n RMS:23.45 um')

fig.colorbar(im, ax=axes.ravel(), orientation='horizontal', location='bottom', label='um', fraction=.05)


"""
fig, axes = plt.subplots(2,2, sharex=1, sharey=1)
im = axes[0,0].pcolormesh(x,y, surf5*mask_plot, vmin=-50, vmax=50, cmap=cm.jet)
plot_deformations(panels, coeffs, ax=axes[0,1], correct_global=False, pol_deg=4)
rms, p = plot_deformations(panels, coeffs, ax=axes[1,1], correct_global=True, pol_deg=4)

axes[1,0].pcolormesh(x,y, surf5_cook*mask_plot, vmin=-50, vmax=50, cmap=cm.jet)

for ax in axes.flatten():
    ax.set_aspect('equal')
    APEX_draw_panels(ax)

axes[0,1].set_title('Kirchhoff-Fresnel optimization\n RMS:34.25um')
axes[0,0].set_title('Classical Fourier method\n RMS:29.19um')
axes[1,0].set_title('Fourier method over optimized beam via Kirchhoff-Fresnel\n RMS:23.45 um')
axes[1,1].set_title('Kirchoff-Fresnel optimization pol order 4 removal \n RMS=%.2f'%(rms*1e6))

fig.colorbar(im, ax=axes.ravel(), orientation='vertical', location='right', label='um', fraction=.05)

for i in range(1,4):
    fig, axes = plt.subplots(2,2, sharex=1, sharey=1)
    im = axes[0,0].pcolormesh(x,y, surf5*mask_plot, vmin=-50, vmax=50, cmap=cm.jet)
    axes[0,0].set_title("Classical Fourier method RMS=29.19 um")
    rms, pol = plot_deformations(panels, coeffs, ax=axes[0,1], correct_global=False, pol_deg=4)
    axes[0,1].set_title("Inferred deformations RMS:%.2f um"%(rms*1e6))
    rms, pol = plot_deformations(panels, coeffs, ax=axes[1,0], correct_global=True, pol_deg=1+i)
    axes[1,0].set_title("Polynomial order %i removal RMS=%.2f um"%((i+1, rms*1e6)))
    axes[1,1].pcolormesh(x,y, pol(x,y)*1e6*mask_plot2, vmin=-50, vmax=50, cmap=cm.jet)
    axes[1,1].set_title("Order %i fit polynomial"%(i+1))
    for ax in axes.flatten():
        APEX_draw_panels(ax)
        ax.set_aspect('equal')
    fig.colorbar(im, ax=axes.ravel(), orientation='vertical', location='right', label='um', fraction=.05)
plt.show()

"""


#fig, axes = plt.subplots(2,2)
#plot_deformations(panels, coeffs, ax=axes[0,0], correct_global=False, pol_deg=2)
#plot_deformations(panels, coeffs, ax=axes[0,1], correct_global=True, pol_deg=2)
#axes[1,0].pcolormesh(x,y, surf5*mask_plot, vmin=-50, vmax=50, cmap=cm.jet)
#axes[1,1].pcolormesh(x,y, surf6*p_mask_plot, vmin=-50, vmax=50, cmap=cm.jet)


"""
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
"""
