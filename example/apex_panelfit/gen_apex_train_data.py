import sys
sys.path.append('../../source')
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
### This code makes a random deformation for the panels in apex and generates
### multiple beam maps with different defocus
###

def pytree_to_numpy(tree):
    return jax.tree_util.tree_map(
            lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x,
            tree
            )

### hyperparameters
##defocus in mm
defocus = [
    [0,0,-15],
    [0,0,0],
    [0,0,15],
    [0,0,7],
    ]
    #[0,0,-15],
    #[0,0,-7],
    #[7,0,15],
    #[7,0,0],
    #[0,7,15],
    #[7,0,0]
    #]

subref_rotations = np.random.randn(3)*10*apu.mdeg
#subref_rotations = np.zeros(3)*10*apu.mdeg
print("rotations: %.4f %.4f %.4f"%(subref_rotations[0].to_value(apu.mdeg),subref_rotations[1].to_value(apu.mdeg),subref_rotations[2].to_value(apu.mdeg)))


train_dir = "train_data_rotation/"

#test
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
#target_points = 513
target_points = 257

## if add the blockage of the legs and secondary
silhouette = True
secondary_silhouette = d2*0.5
sigma_t = 0.002
sigma_r = 0.002

##deformation hyperparameters
key = jax.random.key(np.random.randint(1e5))
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
                                       target_points
                                       )

sec_offsets = jnp.array([x.to_value(apu.m) for x in sec_offsets])
target_pos = jnp.array(target_pos.to_value(apu.m))
s_pos = jnp.array(s_pos.to_value(apu.m))


panels = jax.tree_util.tree_map(jnp.array, panels)
coeffs = generate_start_coeffs(key, panels.keys(), start_rms=start_rms)
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
        #s_pos = s_pos0+sec_offset[None,:]
        s_pos, s_n = secondary_position_update(s_pos0, s_n0, sec_vertex, sec_offsets, sec_rotation)
        p_pos, p_n, p_ds, panel_ms = apply_panel_deformation(panels, coeffs)
        E_i_kf = propagate_cylindrical_gaussian_beam(edge_tapper, horn_aperture, wavel, 
                                                     horn_position, s_pos)
        E_s_kf = kirchhoff_fresnel_scan(s_pos, -s_n, s_ds, E_i_kf, p_pos, wavel, chunk_size=batch_size)
        E_p_k = kirchhoff_fresnel_scan(p_pos, p_n, p_ds, E_s_kf, target_pos, wavel, chunk_size=batch_size)
        return E_p_k
    return forward_function



##iteration changing the values of defocus
forw_function = make_forward_function(panels, s_pos, s_n, s_ds, sec_vertex.to_value(apu.m),
                     target_pos, horn_position, edge_tapper.to_value(apu.dB),
                     horn_aperture.to_value(apu.m), wavel.to_value(apu.m),
                     batch_size)

os.makedirs(os.path.abspath(train_dir), exist_ok=True)
start = time.time()
for defoc in defocus:
    local_start = time.time()
    print(defoc)
    sec_offsets = jnp.array([x*1e-3 for x in defoc])
    E = forw_function(coeffs, sec_offsets, subref_rotations.to_value(apu.rad))
    E_host = jax.block_until_ready(E)
    E_out = E_host.reshape((target_points, target_points))
    name = 'defoc_'+str(defoc[0])+'_'+str(defoc[1])+'_'+str(defoc[2])
    np.savez(os.path.join(os.path.abspath(train_dir),name),
             E = E_out,
             map_size = target_map_size.to_value(apu.deg),
             defocus = defoc
             )
    print("Forward function took: {:.4f}".format((time.time()-local_start)))

print("Train data generation took: {:.4f}".format((time.time()-start)))
print("save coefficients")

coeffs_np = pytree_to_numpy(coeffs)
np.savez(os.path.join(os.path.abspath(train_dir), 'coeffs'),
         coeffs=coeffs_np,
         rotation=subref_rotations.to_value(apu.rad)
         )

