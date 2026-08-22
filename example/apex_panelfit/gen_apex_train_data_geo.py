import sys
sys.path.append('../../source')
from hyperparameters import *
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
defocus =[
    [0,0,-15],
    [0,0,0],
    [0,0,7],
    [0,0,15],
    ]
    
    #[0,0,-15],
    #[0,0,-7],
    #[7,0,15],
    #[7,0,0],
    #[0,7,15],
    #[7,0,0]
    #]

seed = 123
np.random.seed(123)

sec_rotation = np.random.randn(3)*10*apu.mdeg
sec_offsets = np.random.randn(3)*apu.mm

horn_rotation = np.random.randn(3)*10*apu.mdeg
horn_offsets = np.random.randn(3)*apu.mm
horn_offsets[2] = 0*apu.mm

edge_tapper += np.random.randn(1)[0]*0.1*apu.dB
horn_aperture += np.random.randn(1)[0]*0.1*apu.mm


print("sec offsets: %.4f %.4f %.4f"%(sec_offsets[0].to_value(apu.mm),sec_offsets[1].to_value(apu.mm),sec_offsets[2].to_value(apu.mm)))
print("sec rotations: %.4f %.4f %.4f"%(sec_rotation[0].to_value(apu.mdeg),sec_rotation[1].to_value(apu.mdeg),sec_rotation[2].to_value(apu.mdeg)))

print("horn offsets: %.4f %.4f %.4f"%(horn_offsets[0].to_value(apu.mm),horn_offsets[1].to_value(apu.mm),horn_offsets[2].to_value(apu.mm)))
print("horn rotations: %.4f %.4f %.4f"%(horn_rotation[0].to_value(apu.mdeg),horn_rotation[1].to_value(apu.mdeg),horn_rotation[2].to_value(apu.mdeg)))

print("edge tapper:%.4f dB horn aperture:%.4f mm"%(edge_tapper.to_value(apu.dB), horn_aperture.to_value(apu.mm)))


train_dir = "train_data_geo_panels/"
#train_dir = "train_data_geo_ideal/"

#test
os.system("export JAX_ENABLE_X64=True")
jax.config.update('jax_enable_x64', True)

##deformation hyperparameters
key = jax.random.key(np.random.randint(1e5))
start_rms = 60*1e-6                ##this really is mu in a normal distribution

#
batch_size = 256
map_dtype = jnp.complex128

###
###
###

panels, s_pos, s_n, s_ds, sec_vertex, B, target_pos = create_apex_geometries(r_min_prim, d1, r_points,
                      r_min_sec, d2, 
                      t_points_primary, t_points_secondary,
                      f1, f_d, 
                      silhouette, legs_diameter, secondary_silhouette,
                      sigma_t, sigma_r,
                      target_distance, target_map_size, target_points,
                      batch_size=batch_size
                      )


horn_offsets = jnp.array([x.to_value(apu.m) for x in horn_offsets]).astype(jnp.float32)
horn_rotation = jnp.array([x.to_value(apu.rad) for x in horn_rotation]).astype(jnp.float32)
#sec_offsets = jnp.array([x.to_value(apu.m) for x in sec_offsets]).astype(jnp.float32)
sec_rotation = jnp.array([x.to_value(apu.rad) for x in sec_rotation]).astype(jnp.float32)


s_pos = jnp.array(s_pos.to_value(apu.m)).astype(jnp.float32)
##the target positions affect the dynamic range of the output..
target_pos = jnp.array(target_pos.to_value(apu.m)).astype(jnp.float64)
panels = jax.tree_util.tree_map(lambda x: jnp.array(x, dtype=jnp.float32), panels)

coeffs = generate_start_coeffs(key, panels.keys(), start_rms=start_rms, dtype=jnp.float32)
#coeffs = generate_start_coeffs_zeros(panels.keys(), dtype=jnp.float32)

s_n = jnp.array(s_n).astype(jnp.float32)
s_ds = jnp.array(s_ds.to_value(apu.m**2)).astype(jnp.float32)

horn_position = (jnp.array((0,0,B.to_value(apu.m))).T).astype(jnp.float32)
edge_tapper= jnp.array(edge_tapper.to_value(apu.dB))
horn_aperture = jnp.array(horn_aperture.to_value(apu.m))



##before running the eq solver, see the panels
fig, ax = plot_deformations(panels, coeffs)
plt.show()
ans = input('Continue?y/n')
if(ans=='n'):
    sys.exit()


##create the forward function
def make_forward_function(panels, s_pos0, s_n0, s_ds, sec_vertex,
                    target_pos, horn_position, 
                     wavel, batch_size, target_distance, map_dtype=jnp.complex128):
    @jax.jit
    def forward_function(coeffs, 
                         edge_tapper, horn_aperture, 
                         horn_offsets, horn_rotation,
                         sec_offsets, sec_rotation):
        s_pos, s_n = secondary_position_update(s_pos0, s_n0, sec_vertex, sec_offsets, sec_rotation)
        p_pos, p_n, p_ds, deform_ms = apply_panel_deformation(panels, coeffs)

        E_i_kf = propagate_cylindrical_gaussian_beam_offset(edge_tapper, horn_aperture, horn_offsets, 
                                                            horn_rotation, wavel, horn_position, s_pos)
        ##to avoid store intermidiate states, the memory blows up
        E_s_kf = kirchhoff_fresnel_scan_remat(s_pos, -s_n, s_ds, E_i_kf, p_pos, wavel, chunk_size=batch_size, dtype=map_dtype)
        #E_p_k = kirchhoff_fresnel_scan_remat(p_pos, p_n, p_ds, E_s_kf, target_pos, wavel, chunk_size=batch_size, dtype=map_dtype)
        E_p_k = kirchhoff_fresnel_rel_phase_scan_remat(p_pos, p_n, p_ds, E_s_kf, target_pos,
                                                       wavel, pos_ref=target_distance, chunk_size=batch_size, dtype=map_dtype)
        return E_p_k, deform_ms
    return forward_function


forw_function = make_forward_function(panels, s_pos, s_n, s_ds, sec_vertex.to_value(apu.m),
                     target_pos, horn_position, wavel.to_value(apu.m),
                     batch_size, target_distance.to_value(apu.m),
                     map_dtype=map_dtype)



os.makedirs(os.path.abspath(train_dir), exist_ok=True)
start = time.time()
for defoc in defocus:
    local_start = time.time()
    print(defoc)
#sec_offsets = jnp.array([x.to_value(apu.m) for x in sec_offsets]).astype(jnp.float32)
    subref_offsets = jnp.array([sec_offsets[0].to_value(apu.m), sec_offsets[1].to_value(apu.m), defoc[2]*1e-3]).astype(jnp.float32)
    E = forw_function(coeffs, edge_tapper, horn_aperture,
                      horn_offsets, horn_rotation,
                      subref_offsets, sec_rotation)
    E_host, rms = jax.block_until_ready(E)
    E_out = E_host.reshape((target_points, target_points))
    E_norm = E_out/np.max(np.abs(E_out))
    name = 'defoc_'+str(defoc[0])+'_'+str(defoc[1])+'_'+str(defoc[2])
    np.savez(os.path.join(os.path.abspath(train_dir),name),
             E = E_norm,
             map_size = target_map_size.to_value(apu.deg),
             defocus = defoc
             )
    print("Forward function took: {:.4f}".format((time.time()-local_start)))

print("Train data generation took: {:.4f}".format((time.time()-start)))
print("save coefficients")

coeffs_np = pytree_to_numpy(coeffs)
np.savez(os.path.join(os.path.abspath(train_dir), 'coeffs'),
         coeffs=coeffs_np,
         sec_rotation=sec_rotation,
         sec_offsets=sec_offsets.to_value(apu.mm),
         horn_offsets=horn_offsets,
         horn_rotation=horn_rotation,
         edge_tapper=edge_tapper,
         horn_aperture=horn_aperture
         )

