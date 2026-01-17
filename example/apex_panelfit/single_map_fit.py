import sys
sys.path.append('../../source')
from hyperparameters import *
from kirchhoff_fresnel import *
from sources import *
from plot_utils import plot_deformations
import time, os
import jax
from functools import partial
import optax


os.system("export JAX_ENABLE_X64=True")
jax.config.update('jax_enable_x64', True)



learning_rate = 1e-3
sec_offset = [0*apu.mm, 0*apu.mm, 15*apu.mm]


panels, s_pos, s_n, s_ds, B, target_pos = create_apex_geometries(r_min_prim, d1, r_points, 
                      r_min_sec, d2, 
                      t_points_primary, t_points_secondary,
                      f1, f_d, 
                      silhouette, legs_diameter, secondary_silhouette,
                      sigma_t, sigma_r,
                      target_distance, target_map_size, target_points
                      )


#convert the data into jnp arrays

sec_offset = jnp.array([x.to_value(apu.m) for x in sec_offset]).astype(jnp.float32)
s_pos = jnp.array(s_pos.to_value(apu.m)).astype(jnp.float32)
target_pos = jnp.array(target_pos.to_value(apu.m)).astype(jnp.float64)

panels = jax.tree_util.tree_map(lambda x: jnp.array(x, dtype=jnp.float32), panels)
coeffs = generate_start_coeffs(key, panels.keys(), start_rms=start_rms, dtype=jnp.float32)
s_n = jnp.array(s_n).astype(jnp.float32)
s_ds = jnp.array(s_ds.to_value(apu.m**2)).astype(jnp.float32)

horn_position = (jnp.array((0,0,B.to_value(apu.m))).T).astype(jnp.float32)


##create the forward function
def make_forward_function(panels, s_pos0, s_n, s_ds, 
                    target_pos, horn_position, edge_tapper, horn_aperture,
                     wavel, batch_size, map_dtype=jnp.complex64):
    @jax.jit
    def forward_function(coeffs, sec_offset):
        s_pos = s_pos0+sec_offset[None,:]
        p_pos, p_n, p_ds, deform_ms = apply_panel_deformation(panels, coeffs)
        E_i_kf = propagate_cylindrical_gaussian_beam(edge_tapper, horn_aperture, wavel, 
                                                     horn_position, s_pos)
        #E_s_kf = kirchhoff_fresnel_scan(s_pos, -s_n, s_ds, E_i_kf, p_pos, wavel, chunk_size=batch_size)
        #E_p_k = kirchhoff_fresnel_scan(p_pos, p_n, p_ds, E_s_kf, target_pos, wavel, chunk_size=batch_size)
        ##to avoid store intermidiate states
        E_s_kf = kirchhoff_fresnel_scan_remat(s_pos, -s_n, s_ds, E_i_kf, p_pos, wavel, chunk_size=batch_size, dtype=map_dtype)
        E_p_k = kirchhoff_fresnel_scan_remat(p_pos, p_n, p_ds, E_s_kf, target_pos, wavel, chunk_size=batch_size, dtype=map_dtype)
        return E_p_k, deform_ms
    return forward_function


forw_function = make_forward_function(panels, s_pos, s_n, s_ds,
                     target_pos, horn_position, edge_tapper.to_value(apu.dB),
                     horn_aperture.to_value(apu.m), wavel.to_value(apu.m),
                     batch_size)


def loss_funct(coeffs, sec_offset, gold, gamma):
    pred, deform_mse = forw_function(coeffs, sec_offset)
    error = pred-gold
    loss = jnp.mean(error.real**2+error.imag**2)+gamma*jnp.mean(deform_mse)
    return loss



loss_grad = jax.jit(jax.value_and_grad(loss_funct))
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(coeffs)


@jax.jit
def train_step(coeffs, opt_state, sec_offset, gold, gamma):
    loss, grads = loss_grad(coeffs, sec_offset, gold, gamma)
    updates, opt_state = optimizer.update(grads, opt_state, coeffs)
    coeffs = optax.apply_updates(coeffs, updates)
    return coeffs, opt_state, loss




