import sys
sys.path.append('../../source')
from hyperparameters import *
import kirchhoff_fresnel import *
from sources import *
from plot_utils import plot_deformations
import time
import jax
from functools import partial


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

sec_offset = jnp.array([x.to_value(apu.m) for x in sec_offsets])
target_pos = jnp.array(target_pos.to_value(apu.m))
s_pos = jnp.array(s_pos.to_value(apu.m))

panels = jax.tree_util.tree_map(jnp.array, panels)
coeffs = generate_start_coeffs(key, panels.keys(), start_rms=start_rms)
s_n = jnp.array(s_n)
s_ds = jnp.array(s_ds.to_value(apu.m**2))

horn_position = jnp.array((0,0,B.to_value(apu.m))).T


##create the forward function
def make_forward_function(panels, s_pos0, s_n, s_ds, 
                    target_pos, horn_position, edge_tapper, horn_aperture,
                     wavel, batch_size):
    @jax.jit
    def forward_function(coeff, sec_offset):
        s_pos = s_pos0+sec_offset[None,:]
        p_pos, p_n, p_ds, deform_ms = apply_panel_deformation(panels, coeffs)
        E_i_kf = propagate_cylindrical_gaussian_beam(edge_tapper, horn_aperture, wavel, 
                                                     horn_position, s_pos)
        E_s_kf = kirchhoff_fresnel_scan(s_pos, -s_n, s_ds, E_i_kf, p_pos, wavel, chunk_size=batch_size)
        E_p_k = kirchhoff_fresnel_scan(p_pos, p_n, p_ds, E_s_kf, target_pos, wavel, chunk_size=batch_size)
        return E_p_k, deform_mse
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



loss_grad = jax.value_and_grad(loss_funct)
@jax.jit
def train_step(coeffs, opt_state, sec_offset, gold, gamma):
    loss, grads = loss_grad(coeffs, sec_offset, gold, gamma)
    updates, opt_state = optimizer.update(grads, opt_state, coeffs)
    coeffs = optax.apply_updates(coeffs, updates)
    return coeffs, opt_state, loss




