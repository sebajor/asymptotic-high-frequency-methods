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

##deformation hyperparameters
key = jax.random.key(0)
start_rms = 1e-4                ##this really is mu in a normal distribution

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


target_pos = compute_sphere_projection(target_distance, 
                                       target_map_size.to_value(apu.rad),
                                       target_map_size.to_value(apu.rad),
                                       target_points,
                                       target_points
                                       )

sec_offset = jnp.array([x.to_value(apu.m) for x in sec_offset])
target_pos = jnp.array(target_pos.to_value(apu.m))
s_pos = jnp.array(s_pos.to_value(apu.m))


panels = jax.tree_util.tree_map(jnp.array, panels)
coeffs = generate_start_coeffs(key, panels.keys(), start_rms=start_rms)
s_n = jnp.array(s_n.to_value(apu.one))
s_ds = jnp.array(s_ds.to_value(apu.m**2))

horn_position = jnp.array((0,0,B.to_value(apu.m))).T

###ok, here we start the pure jax shit
def forward_function(coeffs, sec_offset, panels, s_pos0, s_n, s_ds, 
                    target_pos, 
                     horn_position, edge_tapper, horn_aperture,
                     wavel,
                     batch_size
                     ):
    s_pos = s_pos0+sec_offset[None,:]
    p_pos, p_n, p_ds = apply_panel_deformation(panels, coeffs)
    E_i_kf = propagate_cylindrical_gaussian_beam(edge_tapper, horn_aperture, wavel, 
                                                 horn_position, s_pos)
    E_s_kf = jax.block_until_ready(kirchhoff_fresnel_scan(s_pos, -s_n, s_ds, E_i_kf, p_pos, wavel, chunk_size=batch_size))
    E_p_k = jax.block_until_ready(kirchhoff_fresnel_scan(p_pos, p_n, p_ds, E_s_kf, target_pos, wavel, chunk_size=batch_size)) 
    return E_p_k


E = forward_function(coeffs, sec_offset, panels, s_pos, s_n, s_ds,
                     target_pos, horn_position, edge_tapper.to_value(apu.dB),
                     horn_aperture.to_value(apu.m), wavel.to_value(apu.m),
                     batch_size)


