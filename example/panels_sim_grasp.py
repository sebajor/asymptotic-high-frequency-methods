import sys
sys.path.append('../source')
from geometry import *
from kirchhoff_fresnel import *
from sources import *
import time
import numpy as np
import os
from apex_geometry import *
from plot_utils import plot_deformations
import jax
from functools import partial

###
### This code makes a random deformation for the panels in apex and computes the
### corresponding beam pattern.
###




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
#sec_offsets = [0*apu.mm, 0*apu.mm, 0*apu.mm]
#sec_offsets = [0*apu.mm, 0*apu.mm, +15*apu.mm]
##nominal holo values
#sec_offsets = [2400*apu.um, -4000*apu.um, 15000*apu.um] 
sec_offsets = [0*apu.um, 0*apu.um, 15000*apu.um] 
#sec_rotation = [-22.7*apu.mdeg, 0.0*apu.deg, 0*apu.deg]   #alpha, beta, gamma
sec_rotation = [0*apu.mdeg, 0.0*apu.deg, 0*apu.deg]   #alpha, beta, gamma



##
target_distance = 1800*apu.m
target_map_size = 3.1*apu.deg     ##size of the map
target_points = 513
endpoint=True

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

###
###
###


pr = np.linspace(r_min_prim, d1/2, r_points)
sr = np.linspace(r_min_sec, d2/2, r_points)
p_tetha = np.linspace(0, 2*np.pi, t_points_primary, endpoint=False)
s_tetha = np.linspace(0, 2*np.pi, t_points_secondary, endpoint=False)

pr_v, pt_v = np.meshgrid(pr, p_tetha) 
sr_v, st_v = np.meshgrid(sr, s_tetha) 


panels, (s_pos, s_n, s_ds, sec_vertex), B, s_focus =  build_apex_model(pr_v, pt_v, sr_v, st_v,
                     primary_focus=f1, f_d=f_d,
                     blockage=silhouette,
                     legs_diameter=legs_diameter,
                     secondary_diameter=secondary_silhouette,
                     sigma_t=sigma_t,
                     sigma_r=sigma_r,
                     batch_size=batch_size
        )


target_pos = compute_sphere_projection(target_distance, 
                                       target_map_size.to_value(apu.rad),
                                       target_map_size.to_value(apu.rad),
                                       target_points,
                                       target_points,
                                       endpoint=endpoint
                                       )

sec_offsets = jnp.array([x.to_value(apu.m) for x in sec_offsets])
sec_rotation = jnp.array([x.to_value(apu.rad) for x in sec_rotation])
target_pos = jnp.array(target_pos.to_value(apu.m))
s_pos = jnp.array(s_pos.to_value(apu.m))


panels = jax.tree_util.tree_map(jnp.array, panels)
#coeffs = generate_start_coeffs(key, panels.keys(), start_rms=start_rms)
coeffs = generate_start_coeffs_zeros(panels.keys(), dtype=jnp.float32)
s_n = jnp.array(s_n)
s_ds = jnp.array(s_ds.to_value(apu.m**2))

horn_position = jnp.array((0,0,B.to_value(apu.m))).T


##before running the eq solver, see the panels
fig, ax = plot_deformations(panels, coeffs)
plt.show()
ans = input('Continue?y/n')
if(ans=='n'):
    sys.exit()


###ok, here we start the pure jax shit
def make_forward_function(panels, s_pos0, s_n0, s_ds, sec_vertex,
                    target_pos, horn_position, edge_tapper, horn_aperture,
                     wavel, batch_size):
    @jax.jit
    def forward_function(coeffs, sec_offsets, sec_rotation):
        #s_pos = s_pos0+sec_offsets[None,:]
        s_pos, s_n = secondary_position_update(s_pos0, s_n0, sec_vertex, sec_offsets, sec_rotation)
        p_pos, p_n, p_ds, panel_ms = apply_panel_deformation(panels, coeffs)
        E_i_kf = propagate_cylindrical_gaussian_beam(edge_tapper, horn_aperture, wavel, 
                                                     horn_position, s_pos)
        E_s_kf = kirchhoff_fresnel_scan(s_pos, -s_n, s_ds, E_i_kf, p_pos, wavel, chunk_size=batch_size)
        #E_p_k = kirchhoff_fresnel_scan(p_pos, p_n, p_ds, E_s_kf, target_pos, wavel, chunk_size=batch_size)
        E_p_k = kirchhoff_fresnel_rel_phase_scan_remat(p_pos, p_n, p_ds, E_s_kf, target_pos, wavel, pos_ref=target_distance, chunk_size=batch_size, dtype=jnp.complex128)
        return E_p_k
    return forward_function


forw_function = make_forward_function(panels, s_pos, s_n, s_ds, sec_vertex.to_value(apu.m),
                     target_pos, horn_position, edge_tapper.to_value(apu.dB),
                     horn_aperture.to_value(apu.m), wavel.to_value(apu.m),
                     batch_size)
start = time.time()
E = forw_function(coeffs, sec_offsets, sec_rotation)
E_host = jax.block_until_ready(E)
print("Forward function took: {:.4f}".format((time.time()-start)))

##plot the resulting beam maps
peak = np.argmax(np.abs(E_host))
E_host = E_host/E_host[peak]
E_out = E_host.reshape((target_points, target_points))
u = np.linspace(-target_map_size/2, target_map_size/2, target_points)
uv, vv = np.meshgrid(u,u)

power = 20*np.log10(np.abs(E_out))
phase = np.rad2deg(np.angle(E_out))

fig, axes = plt.subplots(2,2)
axes[0,0].pcolormesh(uv.to_value(apu.deg), vv.to_value(apu.deg), np.abs(E_out))
axes[0,1].pcolormesh(uv.to_value(apu.deg), vv.to_value(apu.deg), phase)

axes[1,0].plot(u.to_value(apu.deg), power[:, target_points//2], color='darkred')
axes[1,0].plot(u.to_value(apu.deg), np.diag(power), color='darkblue')
axes[1,0].grid()

axes[1,1].plot(u.to_value(apu.deg), phase[:, target_points//2], color='darkred')
axes[1,1].plot(u.to_value(apu.deg), np.diag(phase), color='darkblue')
axes[1,1].grid()

plt.show()

