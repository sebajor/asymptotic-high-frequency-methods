import sys
sys.path.append('../source')
from geometry import *
from kirchhoff_fresnel import *
from sources import *
import time
import numpy as np
import os
from apex_geometry import *

###
### hyperparameters
###

#to have a proper simulation we need the jnp.complex128 representation
os.system("export JAX_ENABLE_X64=True")
jax.config.update('jax_enable_x64', True)


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
r_min_prim = 0.375*apu.m
r_points = 256
t_points_primary = 512      ##angular points for the primary
t_points_secondary = 512    ##angular points for the secondary

##gaussian beam feed horn
edge_tapper = -5*apu.dB
horn_aperture = 3*apu.cm
k_hat = np.array((0,0,1))

##offset of the secondary from nominal position (this should be a parameter, when optimizing)
sec_offsets = [0*apu.mm, 0*apu.mm, 0*apu.mm]
#sec_offsets = [0*apu.mm, 0*apu.mm, +15*apu.mm]

##
target_distance = 1835*apu.m
target_map_size = 3*apu.deg     ##size of the map
target_points = 513

## if add the blockage of the legs and secondary
silhouette = True
secondary_silhouette = d2*0.5
sigma_t = 0.002
sigma_r = 0.002


deformation_coeffs = np.zeros((264, 5)) ##for panels deformation



###
###
###


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

###ok, here we start the pure jax shit

panels = jax.tree_util.tree_map(jnp.array, panels)


##add the defocus to the secondary
s_pos[:,0] += sec_offsets[0]
s_pos[:,1] += sec_offsets[1]
s_pos[:,2] += sec_offsets[2]


##create feed field
source_x0 = np.array((0,0,B.to_value(apu.m))).T*apu.m
source = cylindrical_gaussian_beam(edge_tapper, horn_aperture, 
                                    wavel, source_x0, k_hat)










