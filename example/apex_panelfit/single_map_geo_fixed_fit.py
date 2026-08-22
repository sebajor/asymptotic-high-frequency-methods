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
### This fitting is after the geometrical fitting is done (ie we found the subref and horn positions, rotations
###

os.system("export JAX_ENABLE_X64=True")
jax.config.update('jax_enable_x64', True)



learning_rate = 1e-4#1e-6#1e-3
sec_offsets = [0*apu.mm, 0*apu.mm, 15*apu.mm]
sec_rotation = [0*apu.mdeg, 0*apu.mdeg, 0*apu.mdeg]
#sec_offsets = [0*apu.mm, 0*apu.mm, 0*apu.mm]
##since the values should be in the 1e-6, then the mse will be in 1e-12! 

gamma = 1e4#1e6#4*1e3#5*1e7
gamma2 =0#1e-4
#gold_file = 'train_data_geo_panels/defoc_0_0_15.npz'
#fit_geo_params= 'train_data_geo_panels/fit_geo_params.npz'
gold_file = 'test_phase_corrected.npz'
fit_geo_params = 'fit_geo_params.npz'

iters = 1000#600
map_dtype=jnp.complex128

panels, s_pos, s_n, s_ds, sec_vertex, B, target_pos = create_apex_geometries(r_min_prim, d1, r_points,
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
##read the data from the fit file
geo_params = np.load(fit_geo_params, allow_pickle=1)['params'].tolist()


sec_offsets = jnp.array(geo_params['sec_offsets'], dtype=jnp.float32)
sec_rotation = jnp.array(geo_params['sec_rotation'], dtype=jnp.float32)
horn_offsets = jnp.array(geo_params['horn_offsets'], dtype=jnp.float32)
horn_rotation = jnp.array(geo_params['horn_rotation'], dtype=jnp.float32)
edge_tapper = jnp.array(geo_params['edge_tapper'], dtype=jnp.float32)
horn_aperture = jnp.array(geo_params['horn_aperture'], dtype=jnp.float32)


#convert the data into jnp arrays
s_pos = jnp.array(s_pos.to_value(apu.m)).astype(jnp.float32)
##the target positions affect the dynamic range of the output..
target_pos = jnp.array(target_pos.to_value(apu.m)).astype(jnp.float64)


panels = jax.tree_util.tree_map(lambda x: jnp.array(x, dtype=jnp.float32), panels)
coeffs = generate_start_coeffs(key, panels.keys(), start_rms=start_rms, dtype=jnp.float32)
s_n = jnp.array(s_n).astype(jnp.float32)
s_ds = jnp.array(s_ds.to_value(apu.m**2)).astype(jnp.float32)
horn_position = (jnp.array((0,0,B.to_value(apu.m))).T).astype(jnp.float32)



##create the forward function
def make_forward_function(panels, s_pos0, s_n0, s_ds, sec_vertex,
                    target_pos, horn_position,
                    sec_offsets, sec_rotation,
                    edge_tapper, horn_aperture,
                    horn_offsets, horn_rotation,
                    wavel, batch_size, target_distance, map_dtype=jnp.complex128):
    @jax.jit
    def forward_function(coeffs, sec_adjust):
        sec_offs = sec_offsets+sec_adjust
        s_pos, s_n = secondary_position_update(s_pos0, s_n0, sec_vertex, sec_offs, sec_rotation)
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





##create the forward function
forw_function = make_forward_function(panels, s_pos, s_n, s_ds, sec_vertex.to_value(apu.m),
                     target_pos, horn_position, 
                     sec_offsets, sec_rotation,
                     edge_tapper, horn_aperture,
                     horn_offsets, horn_rotation,
                     wavel.to_value(apu.m),
                     batch_size, target_distance.to_value(apu.m),
                     map_dtype = map_dtype
                     )


#def loss_funct(coeffs, sec_rotation, sec_offsets, gold, gamma):
def loss_funct(params, gold, gamma, gamma2, norm_point=256*128+128):
    pred, deform_mse = forw_function(params['coeffs'], params['sec_adjust']) 
    ##normalize the predictions
    #error = pred-gold
    #pred_norm = pred/jnp.max(jnp.abs(pred))
    pred_norm = pred/pred[norm_point]
    error = pred_norm-gold
    #loss in the aperture
    #pred_reshape = pred.reshape((256,256)) #just to test
    #pred_shift = jnp.fft.ifftshift(pred_reshape)
    #aperture = jnp.fft.fftshift(jnp.fft.ifft2(pred_shift)).flatten()
    #error = aperture-gold
    loss = jnp.mean(error.real**2+error.imag**2)+gamma*jnp.mean(deform_mse)+gamma2*jnp.linalg.norm(params['sec_adjust'])
    #jax.debug.print("jax.debug.print(y) -> {y}", y=jnp.mean(deform_mse))
    return loss

params = {
        "coeffs":coeffs,
        "sec_adjust":jnp.ones(3)*1e-10
        }

loss_grad = jax.jit(jax.value_and_grad(loss_funct))
optimizer = optax.adam(learning_rate)

opt_state = optimizer.init(params)


@jax.jit
def train_step(params, opt_state,  gold, gamma, gamma2):
    loss, grads = loss_grad(params, gold, gamma, gamma2)
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
    params, opt_state, loss = train_step(params, opt_state, E.flatten(), gamma, gamma2)
    losses.append(loss)
    print("iter:%i/t loss:%E/t time:%.3f "%(i, loss, time.time()-start))
    print("sec_adjust %.4f, %.4f, %.4f"%(params['sec_adjust'][0]*1e3,params['sec_adjust'][1]*1e3,params['sec_adjust'][2]*1e3) )
    if((i%10) == 0):
        fig, ax = plt.subplots(2,2,figsize=(10,10))
        plot_deformations(panels, params['coeffs'],ax=ax[0,0], correct_global=0)
        E_pred, deform_mse = forw_function(params['coeffs'], params['sec_adjust'])
        E_pred = np.array(E_pred).reshape((target_points, target_points))
        E_pred = E_pred/E_pred[128,128]
        ax[0,1].plot(20*np.log10(np.abs(np.diag(E_pred))), color='darkblue')
        ax[0,1].plot(20*np.log10(np.abs(np.diag(E))), color='darkred')
        F_shift = np.fft.ifftshift(E_pred)
        ap = np.fft.fftshift(np.fft.ifft2(F_shift))
        ax[1,0].imshow(np.angle(ap))
        ax[1,1].imshow(np.abs(ap))
        title = "iteration "+str(i)+"\n"
        #title += "rotations: %.5f %.5f %.5f"%(
        #        np.rad2deg(params['sec_rotation'][0])*1e3,
        #        np.rad2deg(params['sec_rotation'][1])*1e3,
        #        np.rad2deg(params['sec_rotation'][2]*1e3)
        #            )#+"\n"
        #ax.set_title("iteration "+str(i))
        fig.suptitle(title)
        fig.savefig('images/'+str(i), dpi=100)
        plt.close(fig)

plot_deformations(panels, coeffs)





