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
from hyperparameters import *
import h5py



f = h5py.File("convergence_ouptut.hdf5", "w")
dset = f.create_dataset(
    "outputs",
    shape=(
        target_points,
        target_points,
        len(test_r_points),
        len(test_t_points_p),
        len(test_t_points_s),
    ),
    dtype=np.complex128,
)


sec_offset = [0*apu.mm, 0*apu.mm, 15*apu.mm]

r_points = 256
t_points_primary = 512      ##angular points for the primary
t_points_secondary = 512    ##angular points for the secondary

##The idea of this script is to iterate over sampling to converge
test_r_points = 256+np.arange(15)*52
test_t_points_p = 512+np.arange(10)*52
test_t_points_s = 512+np.arange(10)*52


for i in range(len(test_r_points)):
    for j in range(len(test_t_points_p)):
        for k in range(len(test_t_points_s)):
            r_points = test_r_points[i]
            t_points_primary =  test_t_points_p[j]
            t_points_secondary = test_t_points_s[k]

            panels, s_pos, s_n, s_ds, B, target_pos = create_apex_geometries(r_min_prim, d1, r_points,
                              r_min_sec, d2,
                              t_points_primary, t_points_secondary,
                              f1, f_d,
                              silhouette, legs_diameter, secondary_silhouette,
                              sigma_t, sigma_r,
                              target_distance, target_map_size, target_points,
                              batch_size=batch_size
                              )

            #convert the data into jnp arrays
            sec_offset = jnp.array([x.to_value(apu.m) for x in sec_offset]).astype(jnp.float32)
            s_pos = jnp.array(s_pos.to_value(apu.m)).astype(jnp.float32)
            ##the target positions affect the dynamic range of the output..
            #target_pos = jnp.array(target_pos.to_value(apu.m)).astype(jnp.float64)
            target_pos = jnp.array(target_pos.to_value(apu.m)).astype(jnp.float32)


            panels = jax.tree_util.tree_map(lambda x: jnp.array(x, dtype=jnp.float32), panels)
            coeffs = generate_start_coeffs(key, panels.keys(), start_rms=start_rms, dtype=jnp.float32)
            s_n = jnp.array(s_n).astype(jnp.float32)
            s_ds = jnp.array(s_ds.to_value(apu.m**2)).astype(jnp.float32)

            horn_position = (jnp.array((0,0,B.to_value(apu.m))).T).astype(jnp.float32)

            ###ok, here we start the pure jax shit
            def make_forward_function(panels, s_pos0, s_n, s_ds,
                                target_pos, horn_position, edge_tapper, horn_aperture,
                                 wavel, batch_size):
                @jax.jit
                def forward_function(coeffs, sec_offset):
                    s_pos = s_pos0+sec_offset[None,:]
                    p_pos, p_n, p_ds, panel_ms = apply_panel_deformation(panels, coeffs)
                    E_i_kf = propagate_cylindrical_gaussian_beam(edge_tapper, horn_aperture, wavel,
                                                                 horn_position, s_pos)
                    E_s_kf = kirchhoff_fresnel_scan(s_pos, -s_n, s_ds, E_i_kf, p_pos, wavel, chunk_size=batch_size)
                    E_p_k = kirchhoff_fresnel_scan(p_pos, p_n, p_ds, E_s_kf, target_pos, wavel, chunk_size=batch_size)
                    return E_p_k
                return forward_function
            forw_function = make_forward_function(panels, s_pos, s_n, s_ds,
                     target_pos, horn_position, edge_tapper.to_value(apu.dB),
                     horn_aperture.to_value(apu.m), wavel.to_value(apu.m),
                     batch_size)
            E = forw_function(coeffs, sec_offset)
            E_host = jax.block_until_ready(E)
            E_out = E_host.reshape((target_points, target_points))
            dset[:,:,i,j,k] = np.array(E_out)
            f.flush()
f.close()

