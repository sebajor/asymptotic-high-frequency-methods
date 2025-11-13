import sys
sys.path.append('../source')
from geometry import *
from kirchhoff_fresnel import *
from sources import *
import time
import numpy as np
import os

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

r_min = 0.001*apu.m
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


## 
batch_size = 256

###
###
###

##get geometry
pr = np.linspace(r_min, d1/2, r_points)
sr = np.linspace(r_min, d2/2, r_points)
p_tetha = np.linspace(0, 2*np.pi, t_points_primary, endpoint=False)
s_tetha = np.linspace(0, 2*np.pi, t_points_secondary, endpoint=False)

pr_v, pt_v = np.meshgrid(pr, p_tetha) 
sr_v, st_v = np.meshgrid(sr, s_tetha) 

(p_pos, p_n, p_ds), (s_pos, s_n, s_ds), B, s_focus = cassegrain_cylindrical_cone(pr_v, pt_v, sr_v, st_v,
                           f1, f_d)


##add the defocus to the secondary
s_pos[:,0] += sec_offsets[0]
s_pos[:,1] += sec_offsets[1]
s_pos[:,2] += sec_offsets[2]


##create feed field
source_x0 = np.array((0,0,B.to_value(apu.m))).T*apu.m
source = cylindrical_gaussian_beam(edge_tapper, horn_aperture, 
                                    wavel, source_x0, k_hat)
##propagate the gaussian beam to the secondary
E_i_kf = source.propagate(s_pos)

###get target positions
target_pos = compute_sphere_projection(target_distance, 
                                       target_map_size.to_value(apu.rad),
                                       target_map_size.to_value(apu.rad),
                                       target_points,
                                       target_points
                                       )

### generate mask to emulate the blockage.. only if silhouette=True
if(silhouette):
    """
    ##this is the most basic mask..since it has strong discontinuities generates
    ##sinc like structures
    #mask = np.ones(p_pos.shape[0], dtype=bool)
    #r_xy = np.sqrt(p_pos[:,0]**2+p_pos[:,1]**2)
    #mask[r_xy<d2.to_value(apu.m)*0.5] = False
    #mask[np.abs(p_pos[:,0])<legs_diameter.to_value(apu.m)/2] = False
    #mask[np.abs(p_pos[:,1])<legs_diameter.to_value(apu.m)/2] = False
    """
    mask = cassegrain_silhouettes(p_pos, legs_diameter=legs_diameter, 
            secondary_diameter=secondary_silhouette,
            sigma_t=sigma_t, sigma_r=sigma_r)
    mask = jnp.array(mask)
else:
    mask = jnp.ones(p_pos.shape[0])


### convert the parameters into jnp avoiding units
##after checking the only object that needs to be f64 is the target_positions
s_pos = jnp.array(s_pos.to_value(apu.m)).astype(jnp.float32)
s_n = jnp.array(s_n).astype(jnp.float32)
s_ds = jnp.array(s_ds.to_value(apu.m**2)).astype(jnp.float32)
E_i_kf = jnp.array(E_i_kf.to_value(apu.V/apu.m)).astype(jnp.float32)
wavel = jnp.float32(wavel.to_value(apu.m))
p_pos = jnp.array(p_pos.to_value(apu.m)).astype(jnp.float32)
p_n = jnp.array(p_n).astype(jnp.float32)
p_ds = jnp.array(p_ds.to_value(apu.m**2)).astype(jnp.float32)
target_pos = jnp.array(target_pos.to_value(apu.m)).astype(jnp.float32)

####
start = time.time()
print("Starting computation")
E_s_kf = jax.block_until_ready(kirchhoff_fresnel_scan(s_pos, -s_n, s_ds, E_i_kf, p_pos, wavel, chunk_size=batch_size))
print("Integration over secondary done")
E_s_kf_block = E_s_kf*mask

E_p_k = jax.block_until_ready(kirchhoff_fresnel_scan(p_pos, p_n, p_ds, E_s_kf_block, target_pos, wavel, chunk_size=batch_size)) 
print("Integration over primary done")

print("Kirchhoff-Fresnel integration took %.4f"%(time.time()-start))


###generate some plots

E_p_k = E_p_k.reshape((target_points, target_points))
pow_kf_db = 20*np.log10(np.abs(E_p_k))
phase_kf = np.rad2deg(np.angle(E_p_k))

u = np.linspace(-target_map_size/2, target_map_size/2, target_points)
uv, vv = np.meshgrid(u,u)


fig, axes = plt.subplots(2,2)
axes[0,0].pcolormesh(uv.to_value(apu.deg), vv.to_value(apu.deg), pow_kf_db)
axes[0,1].pcolormesh(uv.to_value(apu.deg), vv.to_value(apu.deg), phase_kf)

axes[1,0].plot(u.to_value(apu.deg), np.diag(pow_kf_db)-np.max(pow_kf_db))
axes[1,1].plot(u.to_value(apu.deg), np.diag(phase_kf))

axes[0,0].set_title("Beam map dB")
axes[0,1].set_title("Beam map phase deg")
axes[1,0].set_title("45 cut dB")
axes[1,1].set_title("45 cut deg")

axes[1,0].grid(); axes[1,1].grid()
plt.show()

