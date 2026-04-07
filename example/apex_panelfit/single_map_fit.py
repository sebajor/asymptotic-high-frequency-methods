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


#os.system("export JAX_ENABLE_X64=True")
#jax.config.update('jax_enable_x64', True)



learning_rate = 1e-6#1e-3
#sec_offset = [0*apu.mm, 0*apu.mm, 15*apu.mm]
sec_offset = [0*apu.mm, 0*apu.mm, 0*apu.mm]
##since the values should be in the 1e-6, then the mse will be in 1e-12! 
#gamma = 5*1e10
gamma = 5*1e7
gold_file = "train_data/defoc_0_0_0.npz"    ##this is a fake data generated

iters = 600

panels, s_pos, s_n, s_ds, B, target_pos = create_apex_geometries(r_min_prim, d1, r_points, 
                      r_min_sec, d2, 
                      t_points_primary, t_points_secondary,
                      f1, f_d, 
                      silhouette, legs_diameter, secondary_silhouette,
                      sigma_t, sigma_r,
                      target_distance, target_map_size, target_points,
                      batch_size=batch_size
                      )
##now if the panels points does not match with the batch_size there will be points out so I add some fake points
"""
panels_points = 0
for k,p in panels.items():
    panels_points += p['p0'].shape[0]
if(panels_points%batch_size != 0):
    remain = batch_size-panels_points%batch_size
    panels['fake'] = {
            'p0'        : np.random.random((remain, 3)),
            'n0'        : np.random.random((remain, 3)),
            'ds0'       : np.zeros(remain),
            's_0r'      : np.random.random((remain, 3)),
            's_0t'      : np.random.random((remain, 3)),
            'dn_dr'     : np.random.random((remain, 3)),
            'dn_dt'     : np.random.random((remain, 3)),
            'x_'        : np.random.random(remain),
            'y_'        : np.random.random(remain),
            'cte_sr'    : np.random.random(remain),
            'cte1_st'   : np.random.random(remain),
            'cte2_st'   : np.random.random(remain),
            'r'         : np.random.random(remain),
            'blockage'  : np.zeros(remain)
            }
"""

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


##create the forward function
def make_forward_function(panels, s_pos0, s_n, s_ds, 
                    target_pos, horn_position, edge_tapper, horn_aperture,
                     wavel, batch_size, target_distance, map_dtype=jnp.complex64):
    @jax.jit
    def forward_function(coeffs, sec_offset):
        s_pos = s_pos0+sec_offset[None,:]
        p_pos, p_n, p_ds, deform_ms = apply_panel_deformation(panels, coeffs)
        E_i_kf = propagate_cylindrical_gaussian_beam(edge_tapper, horn_aperture, wavel, 
                                                     horn_position, s_pos)
        #E_s_kf = kirchhoff_fresnel_scan(s_pos, -s_n, s_ds, E_i_kf, p_pos, wavel, chunk_size=batch_size)
        #E_p_k = kirchhoff_fresnel_scan(p_pos, p_n, p_ds, E_s_kf, target_pos, wavel, chunk_size=batch_size)

        ##to avoid store intermidiate states, the memory blows up
        E_s_kf = kirchhoff_fresnel_scan_remat(s_pos, -s_n, s_ds, E_i_kf, p_pos, wavel, chunk_size=batch_size, dtype=map_dtype)
        #E_p_k = kirchhoff_fresnel_scan_remat(p_pos, p_n, p_ds, E_s_kf, target_pos, wavel, chunk_size=batch_size, dtype=map_dtype)
        E_p_k = kirchhoff_fresnel_rel_phase_scan_remat(p_pos, p_n, p_ds, E_s_kf, target_pos, wavel, pos_ref=target_distance, chunk_size=batch_size, dtype=map_dtype)
        return E_p_k, deform_ms
    return forward_function


forw_function = make_forward_function(panels, s_pos, s_n, s_ds,
                     target_pos, horn_position, edge_tapper.to_value(apu.dB),
                     horn_aperture.to_value(apu.m), wavel.to_value(apu.m),
                     batch_size, target_distance.to_value(apu.m))


def loss_funct(coeffs, sec_offset, gold, gamma):
    pred, deform_mse = forw_function(coeffs, sec_offset)
    ##normalize the predictions
    error = pred-gold
    #loss in the aperture
    #pred_reshape = pred.reshape((256,256)) #just to test
    #pred_shift = jnp.fft.ifftshift(pred_reshape)
    #aperture = jnp.fft.fftshift(jnp.fft.ifft2(pred_shift)).flatten()
    #error = aperture-gold
    loss = jnp.mean(error.real**2+error.imag**2)+gamma*jnp.mean(deform_mse)
    #jax.debug.print("jax.debug.print(y) -> {y}", y=jnp.mean(deform_mse))
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


print("Training loop")
f = np.load(gold_file, allow_pickle=1)
E = jnp.array(f['E'])
losses = []

train_request = input("Do you want to start the train?")
if(train_request!= 'y'):
    sys.exit()


for i in range(iters):
    start = time.time()
    coeffs, opt_state, loss = train_step(coeffs, opt_state, sec_offset,E.flatten(), gamma)
    losses.append(loss)
    print("iter:%i/t loss:%.3f/t time:%.3f "%(i, loss, time.time()-start))
    if((i%10) == 0):
        fig, ax = plt.subplots(figsize=(10,10))
        plot_deformations(panels, coeffs,ax=ax, correct_global=0)
        ax.set_title("iteration "+str(i))
        fig.savefig('images/'+str(i), dpi=100)
        plt.close(fig)

plot_deformations(panels, coeffs)





