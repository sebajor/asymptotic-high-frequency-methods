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


###
### This code does a full fitting over the 6 parameters of the subreflector.. It was  shown that it doesnt work :(
###


os.system("export JAX_ENABLE_X64=True")
jax.config.update('jax_enable_x64', True)



learning_rate = 1e-4 #1e-6#1e-3
sec_offsets = [0*apu.mm, 0*apu.mm, 15*apu.mm]
##since the values should be in the 1e-6, then the mse will be in 1e-12! 
gamma = 1#5*1e7
gold_file = 'test_phase_corrected.npz'

iters = 1000#600
map_dtype = jnp.complex128

panels, s_pos, s_n, s_ds, sec_vertex, B, target_pos = create_apex_geometries(r_min_prim, d1, r_points,
                      r_min_sec, d2, 
                      t_points_primary, t_points_secondary,
                      f1, f_d, 
                      silhouette, legs_diameter, secondary_silhouette,
                      sigma_t, sigma_r,
                      target_distance, target_map_size, target_points,
                      batch_size=batch_size
                      )


#convert the data into jnp arrays
horn_offsets = jnp.array([x.to_value(apu.m) for x in horn_offsets]).astype(jnp.float32)
horn_rotation = jnp.array([x.to_value(apu.rad) for x in horn_rotation]).astype(jnp.float32)
sec_offsets = jnp.array([x.to_value(apu.m) for x in sec_offsets]).astype(jnp.float32)
sec_rotation = jnp.array([x.to_value(apu.rad) for x in sec_rotation]).astype(jnp.float32)


s_pos = jnp.array(s_pos.to_value(apu.m)).astype(jnp.float32)
##the target positions affect the dynamic range of the output..
target_pos = jnp.array(target_pos.to_value(apu.m)).astype(jnp.float64)
panels = jax.tree_util.tree_map(lambda x: jnp.array(x, dtype=jnp.float32), panels)

##for this specific code this should be all zeros!
coeffs = generate_start_coeffs_zero(panels.keys(), dtype=jnp.float32)
s_n = jnp.array(s_n).astype(jnp.float32)
s_ds = jnp.array(s_ds.to_value(apu.m**2)).astype(jnp.float32)

horn_position = (jnp.array((0,0,B.to_value(apu.m))).T).astype(jnp.float32)
edge_tapper= jnp.array(edge_tapper.to_value(apu.dB))
horn_aperture = jnp.array(horn_aperture.to_value(apu.m)))


##create the forward function
def make_forward_function(panels, s_pos0, s_n0, s_ds, sec_vertex,
                    target_pos, horn_position, 
                     wavel, batch_size, target_distance, map_dtype=jnp.complex128):
    @jax.jit
    def forward_function(coeffs, 
                         edge_tapper, horn_aperture, 
                         horn_offsets, horn_rotation,
                         sec_rotation, sec_offsets):
        s_pos, s_n = secondary_position_update(s_pos0, s_n0, sec_vertex, sec_offsets, sec_rotation)
        p_pos, p_n, p_ds, deform_ms = apply_panel_deformation(panels, coeffs)

        E_i_kf = propagate_cylindrical_gaussian_beam_offset(edge_tapper, horn_aperture, horn_offsets, 
                                                            horn_rotation, wavel, horn_posiion, s_pos)
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


#def loss_funct(coeffs, sec_rotation, sec_offsets, gold, gamma):
def loss_funct(params, gold, gamma):
    pred, deform_mse = forw_function(params['coeffs'], 
                                     params['edge_tapper'], params['horn_aperture'], 
                                     params['horn_offsets'], params['horn_rotation'],
                                     params['sec_rotation'], params['sec_offsets'])
    pred_norm = pred/jnp.max(jnp.abs(pred))
    error = pred_norm-gold
    loss = jnp.mean(error.real**2+error.imag**2)+gamma*jnp.mean(deform_mse)
    #jax.debug.print("jax.debug.print(y) -> {y}", y=jnp.mean(deform_mse))
    return loss

params = {
        "coeffs":coeffs,
        "sec_rotation": sec_rotation,
        "sec_offsets": sec_offsets,
        'edge_tapper': edge_tapper,
        'horn_aperture': horn_aperture,
        'horn_offset': horn_offset,
        'horn_rotation': horn_rotation
        }

loss_grad = jax.jit(jax.value_and_grad(loss_funct))
optimizer = optax.adam(learning_rate)

opt_state = optimizer.init(params)


@jax.jit
def train_step(params, opt_state, gold, gamma):
    loss, grads = loss_grad(params, gold, gamma)
    updates, opt_state = optimizer.update(grads, opt_state, params) ##sec_rotation (?)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


print("Training loop")
f = np.load(gold_file, allow_pickle=1)
E = jnp.array(f['E'])
losses = []

train_request = input("Do you want to start the train?")
if(train_request!= 'y'):
    sys.exit()


for i in range(iters):
    start = time.time()
    params, opt_state, loss = train_step(params, opt_state, E.flatten(), gamma)
    losses.append(loss)
    print("iter:%i \t loss:%E \t time:%.3f "%(i, loss, time.time()-start))
    if((i%10) == 0):
        fig, ax = plt.subplots(2,2,figsize=(10,10))
        plot_deformations(panels, params['coeffs'],ax=ax[0,0], correct_global=0)
        E_pred, deform_mse = forw_function(params['coeffs'], params['sec_rotation'], params['sec_offsets'])
        E_pred = np.array(E_pred).reshape((target_points, target_points))
        E_pred = E_pred/np.max(np.abs(E_pred))
        ax[0,1].plot(20*np.log10(np.abs(np.diag(E_pred))), color='darkblue')
        ax[0,1].plot(20*np.log10(np.abs(np.diag(E))), color='darkred')
        F_shift = np.fft.ifftshift(E_pred)
        ap = np.fft.fftshift(np.fft.ifft2(F_shift))
        ax[1,0].imshow(np.angle(ap))
        ax[1,1].imshow(np.abs(ap))
        title = "iteration "+str(i)+"\n"
        title += "suref_pos:%.5f %.5f %.5f"%(params['sec_offsets'][0]*1e3, 
                                   params['sec_offsets'][1]*1e3,
                                   params['sec_offsets'][2]*1e3)+"\n"

        title += "subref_rotation: %.5f %.5f %.5f"%(
                np.rad2deg(params['sec_rotation'][0])*1e3,
                np.rad2deg(params['sec_rotation'][1])*1e3,
                np.rad2deg(params['sec_rotation'][2])*1e3)+"\n"
        title += "horn pos: %.5f %.5f %.5f"%(params['horn_offsets'][0]*1e3,
                                             params['horn_offsets'][1]*1e3,
                                             params['horn_offsets'][2]*1e3
                )+"\n"
        title += "horn rotation: %.5f %.5f %.5f"%(np.rad2deg(params['horn_rotation'][0])*1e3,
                                             np.rad2deg(params['horn_rotation'][1])*1e3,
                                             np.rad2deg(params['horn_rotation'][2])*1e3
                )
        #ax.set_title("iteration "+str(i))
        fig.suptitle(title)
        fig.savefig('images/'+str(i), dpi=100)
        plt.close(fig)

plot_deformations(panels, coeffs)





