import sys
sys.path.append('../../source')
import astropy.units as apu
import astropy.constants as cte
import numpy as np
from geometry import *
from apex_geometry import *

##
freq = 92.4*apu.GHz
wavel = cte.c/freq

###cassegrain geometry
d1 = 12*apu.m
d2 = 0.75*apu.m
f1 = 4.8*apu.m
f_d = 8
s = 1.05        ##oversize of the secondary
legs_diameter = 0.05*apu.m#0.1*apu.m

r_min_sec = 0.001*apu.m
r_min_prim = 0.001*apu.m#0.375*apu.m    ##there is something weird in the blockage calculation if rmin
                                        ##is big.. some nans start to pop up... 
                                        ##having a big value here causes that several
                                        ##points are wasted in a part that has no panels..
r_points = 256
t_points_primary = 512      ##angular points for the primary
t_points_secondary = 512    ##angular points for the secondary

##gaussian beam feed horn
edge_tapper = -5*apu.dB
horn_aperture = 3*apu.cm
k_hat = np.array((0,0,1))

##offset of the secondary from nominal position (this should be a parameter, when optimizing)
sec_offset = [0*apu.mm, 0*apu.mm, 0*apu.mm]
#sec_offset = [0*apu.mm, 0*apu.mm, +15*apu.mm]

##
target_distance = 1835*apu.m
target_map_size = 3*apu.deg     ##size of the map
target_points = 513

## if add the blockage of the legs and secondary
silhouette = True
secondary_silhouette = d2*0.5
sigma_t = 0.002
sigma_r = 0.002

##deformation hyperparameters
key = jax.random.key(0)
start_rms = 60*1e-6                ##this really is mu in a normal distribution

#
batch_size = 256


def create_apex_geometries(r_min_prim, d1, r_points, 
                      r_min_sec, d2, 
                      t_points_primary, t_points_secondary,
                      f1, f_d, 
                      silhouette, legs_diameter, secondary_silhouette,
                      sigma_t, sigma_r,
                      target_distance, target_map_size, target_points
                      ):
    pr = np.linspace(r_min_prim, d1/2, r_points)
    sr = np.linspace(r_min_sec, d2/2, r_points)
    p_tetha = np.linspace(0, 2*np.pi, t_points_primary, endpoint=False)
    s_tetha = np.linspace(0, 2*np.pi, t_points_secondary, endpoint=False)

    pr_v, pt_v = np.meshgrid(pr, p_tetha) 
    sr_v, st_v = np.meshgrid(sr, s_tetha) 


    panels, (s_pos, s_n, s_ds), B, s_focus =  build_apex_model(pr_v, pt_v, sr_v, st_v, 
                     primary_focus=f1, f_d=f_d,
                     blockage=silhouette,
                     legs_diameter=legs_diameter,
                     secondary_diameter=secondary_silhouette,
                     sigma_t=sigma_t,
                     sigma_r=sigma_r
        )


    target_pos = compute_sphere_projection(target_distance, 
                                       target_map_size.to_value(apu.rad),
                                       target_map_size.to_value(apu.rad),
                                       target_points,
                                       target_points
                                       )
    return panels, s_pos, s_n, s_ds, B, target_pos

