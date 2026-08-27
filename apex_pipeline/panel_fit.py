import sys
sys.path.append('../source')
from hyperparameters import *
from kirchhoff_fresnel import *
from sources import *
from plot_utils import plot_deformations
import time, os
import jax
from functools import partial
import optax
import argparse
import h5py
import logging
import apex_utils
import signal

###
### This code does the geometrical fitting of the measured beam pattern.
### The optimization has a limited number of iterations that you can set and also
### a convergence limit (when the update in the loss is less than the convergence 
### limit the optimization stops) and a lower limit on the loss to stop the optimization
###


##set the 64 bit operations at the GPU (we need complex128 to resolve the phase
##at the SNR that we want)
os.system("export JAX_ENABLE_X64=True")
jax.config.update('jax_enable_x64', True)


###
###NOTE: the map is by default inverted, bcs of the sampling we cannot just do F=F[::-1, ::-1] 
###
pol_deg =4


parser = argparse.ArgumentParser(
    description="Geometrical fit for apex hologrpahy pipeline")

parser.add_argument('-f', '--filename', dest='filename' , type=str,
                    help="Input filename should be a .reg or .hdf5 output of the fourier pipeline")
parser.add_argument('-lr', '--learn_rate', dest='learning_rate', type=float, default=1e-4, 
                    help="learning rate for the optimizer")
parser.add_argument('-mi', '--max_iters', dest='max_iters', type=int, default=1000,
                    help="maximum iterations of the optimization")
parser.add_argument('-cl', '--conv_lim', dest='conv_lim', type=float, default=1*1e-11,
                    help="convergence limit")
parser.add_argument("-ll", "--loss_lim", dest='loss_lim', type=float, default=2.1*1e-6,
                    help="lower limit of the loss")

parser.add_argument("-no_stop", "--no_stop", dest="no_stop", action='store_true',
                    help="Avoids all the stop mechanism, the optimization runs up to the max iteration")

parser.add_argument("-gamma", "--gamma", dest='gamma', type=float, default=10,
                    help="lagrange multiplier for the surface error")

parser.add_argument("-geo_file", "--geo_file", dest='geo_file', type=str, default=None,
                    help="Geometry file. If None and you set a hdf5 file as filename the code checks for it in the standard location")

parser.add_argument("-panel_file", "--panel_file", dest='panel_file', type=str, default=None,
                    help="Panel deformation file (.npz). If None just start the coefficients at random")

parser.add_argument("-pi","--plot_interval",dest='plot_interval', type=int, default=10,
                   help="How often generate the debugging plots")
parser.add_argument("-plot_path", dest='plot_path', type=str, default="~/MODULES/physical_optics/")

###
###
###
args = parser.parse_args()

def phase_correction(F,uv,vv,wavel,d1=2.18*apu.m,d2=7.485*apu.m):
    """
    Phase compensation. This function used to be in the regrid code but for easy
    reading was moved here. We are not 100% sure about it, but we guess that 
    cames from the pathlength difference between the main and the reference 
    receivers... Also we dont know what actually are the distances d1 and d2.
    """
    tetha = np.sqrt(uv**2+vv**2)
    correction = 2*np.pi/wavel*(d1-d2)*(1-np.cos(tetha.to_value(apu.rad)))*apu.rad
    phase = np.angle(F)*apu.rad
    phase += correction
    amp = np.abs(F)
    out = amp*np.exp(1j*phase.to_value(apu.rad))
    return out


plot_dir = os.path.expanduser(args.plot_path)
os.makedirs(plot_dir, exist_ok=True)

filepath = os.path.abspath(os.path.expanduser(args.filename))
filename = os.path.basename(filepath)
if(filename.endswith('.reg')):
    u,v,amp,phase = np.loadtxt(filepath)
    N = int(np.sqrt(len(amp)))
    F = (amp*np.exp(1j*phase)).reshape((N,N))
    u = -u.reshape((N,N))*apu.deg
    v = -v.reshape((N,N))*apu.deg
    ##phase correction
    F = phase_correction(F,u,v,wavel)
    F = F/F[128,128]
    F = np.conjugate(F)
    plot_path = os.path.join(plot_dir, filename.split('.reg')[0])

elif(filename.endswith('.hdf5')):
    f = h5py.File(filepath, 'r')
    F = np.array(f['phase_correction']['data'])
    F = F/F[128,128]
    F = np.conjugate(F)
    plot_path = os.path.join(plot_dir, os.path.split(os.path.split(os.path.split(filepath)[0])[0])[1])

else:
    print("Unsuported input file.. must be .reg or .hdf5!")
    sys.exit(1)

###get the geometry parameters
if((args.geo_file is None) and filename.endswith('.hdf5')):
    print("geo_file not set, looking for the last geometrical fit..")
    #geo_path = os.path.join(os.path.split(os.path.split(filepath)[0])[0], 
    #                        "geometry_fit", "geo_params.npz")
    geo_path = os.path.join(plot_path,
                            "geometry_fit")#, "geo_params.npz")
    dirs = os.listdir(geo_path)
    geo_path = os.path.join(geo_path, str(len(dirs)), "geo_params.npz")
    if(not os.path.exists(geo_path)):
        print("Gemetry file not found at %s"%geo_path)
        sys.exit(1)
elif(not os.path.exists(os.path.expanduser(args.geo_file))):
    print("Geometry file not given!")

else:
    geo_path = os.path.expanduser(args.geo_file)

geo_params = np.load(geo_path, allow_pickle=1)['params'].tolist()
print("Using geometry parameters from %s"%geo_params)


print("Creating the plot directories at %s"%plot_path)
os.makedirs(plot_path, exist_ok=True)
plot_path = os.path.join(plot_path, 'panel_fit')
os.makedirs(plot_path, exist_ok=True)

###check if exists another iteration
dirs = os.listdir(plot_path)
plot_path = os.path.join(plot_path, "%03d"%(len(dirs)))
os.makedirs(plot_path, exist_ok=True)


####


learning_rate = args.learning_rate
gamma = args.gamma
iters = args.max_iters
loss_lim = args.loss_lim
conv_lim = args.conv_lim
plot_interval = args.plot_interval

map_dtype = jnp.complex128

###
### Here we start to define everything! Note that most of the hyperparams are in 
### in the hyperparameter file!
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



sec_offsets = jnp.array(geo_params['sec_offsets'], dtype=jnp.float32)
sec_rotation = jnp.array(geo_params['sec_rotation'], dtype=jnp.float32)
horn_offsets = jnp.array(geo_params['horn_offsets'], dtype=jnp.float32)
horn_rotation = jnp.array(geo_params['horn_rotation'], dtype=jnp.float32)
edge_tapper = jnp.array(geo_params['edge_tapper'], dtype=jnp.float32)
horn_aperture = jnp.array(geo_params['horn_aperture'], dtype=jnp.float32)


#convert the data into jnp arrays
s_pos = jnp.array(s_pos.to_value(apu.m)).astype(jnp.float32)
s_n = jnp.array(s_n).astype(jnp.float32)
s_ds = jnp.array(s_ds.to_value(apu.m**2)).astype(jnp.float32)
horn_position = (jnp.array((0,0,B.to_value(apu.m))).T).astype(jnp.float32)
##the target positions affect the dynamic range of the output..
target_pos = jnp.array(target_pos.to_value(apu.m)).astype(jnp.float64)
panels = jax.tree_util.tree_map(lambda x: jnp.array(x, dtype=jnp.float32), panels)

if(args.panel_file is None):
    print("Using random coefficients for panel deformations")
    coeffs = generate_start_coeffs(key, panels.keys(), start_rms=start_rms, dtype=jnp.float32)
else:
    panel_path= os.path.abspath(os.path.expanduser(args.panel_file))
    panel_path= os.path.basename(panel_path)
    panel = np.load(panel_path, allow_pickle=1)
    print("Importing coeff deformations from %s"%panel_path)
    coeffs = panel['params'].tolist()['coeffs']
    print("Done")




##create the forward function
def make_forward_function(panels, s_pos0, s_n0, s_ds, sec_vertex,
                    target_pos, horn_position,
                    sec_offsets, sec_rotation,
                    edge_tapper, horn_aperture,
                    horn_offsets, horn_rotation,
                    wavel, batch_size, target_distance, map_dtype=jnp.complex128):
    @jax.jit
    def forward_function(coeffs):
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
def loss_funct(params, gold, gamma, norm_point=256*128+128):
    pred, deform_mse = forw_function(params['coeffs'])
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
    loss = jnp.mean(error.real**2+error.imag**2)+gamma*jnp.mean(deform_mse)
    #jax.debug.print("jax.debug.print(y) -> {y}", y=jnp.mean(deform_mse))
    return loss

params = {
        "coeffs":coeffs
        }

loss_grad = jax.jit(jax.value_and_grad(loss_funct))
optimizer = optax.adam(learning_rate)

opt_state = optimizer.init(params)


@jax.jit
def train_step(params, opt_state,  gold, gamma):
    loss, grads = loss_grad(params, gold, gamma)
    updates, opt_state = optimizer.update(grads, opt_state, params) ##sec_rotation (?)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

###
###
###
def plot_debug(params, F, target_points, plot_path, iteration):
    fig, ax = plt.subplots(2,2,figsize=(10,10))
    plot_deformations(panels, params['coeffs'],ax=ax[0,0], correct_global=0)

    pred, deform_mse = forw_function(params['coeffs'])
    F_pred = pred/pred[257*128]
    F_pred = np.array(F_pred).reshape((target_points, target_points))
    error = F-F_pred
    beam_error = np.mean(error.real**2+error.imag**2)
    surf_error = np.mean(deform_mse)

    ax[0,1].plot(20*np.log10(np.abs(np.diag(F_pred))), color='darkblue')
    ax[0,1].plot(20*np.log10(np.abs(np.diag(F))), color='darkred')
    ax[0,1].set_ylim(-85, 5)

    E_shift = np.fft.ifftshift(F_pred)
    aperture = np.fft.fftshift(np.fft.ifft2(E_shift))
    ax[1,0].imshow(np.angle(aperture))
    ax[1,1].imshow(np.abs(aperture))
    title = "iteration "+str(iteration)+"\n"
    title += "beam error: %E  surf error:%E"%(beam_error, surf_error)
    fig.suptitle(title)
    fig.savefig(os.path.join(plot_path, str(i)))
    plt.close(fig)


def pytree_to_numpy(tree):
    return jax.tree_util.tree_map(
            lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x,
            tree)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
    losses = []
    print("Starting Panel fitting")
    for i in range(args.max_iters):
        start = time.time()
        params, opt_state, loss = train_step(params, opt_state, F.flatten(), gamma)
        losses.append(loss)
        log_msg = "iter:%i \t loss:%E \t time:%.3f "%(i, loss, time.time()-start)
        logging.info(log_msg)
        if(i%plot_interval==0):
            plot_debug(params, F, target_points, plot_path, i)
        if(not args.no_stop):
            if(np.mean(losses[-10:]) < loss_lim):
                print("Loss limit reached at iteration %i: %E < %E"%(i, loss, loss_lim))
                break
            if(np.mean(np.abs(np.diff(losses[-10:])))< conv_lim):
                print("Loss convergence reached at iteration %i: %E < %E"%(i, np.mean(np.abs(np.diff(losses[-10:]))), conv_lim))
                break
    print("Out if the optimization loop")
    out_params = pytree_to_numpy(params)
    pred, deform_mse = forw_function(params['coeffs'])
    F_pred = pred/pred[257*128]
    F_pred = np.array(F_pred).reshape((target_points, target_points))
    unflip_params = apex_utils.flip_panels_coeffs(out_params['coeffs'])


    pol = large_scale_fitting(panels, unflip_params, pol_deg)
    np.savez(os.path.join(plot_path, "panels_params.npz"),
            flip_params=out_params,
            params = unflip_params,
            losses=losses,
            gamma= gamma,
            F_pred=F_pred,
            F_gold=F,
            large_scale_coeffs = pol.parameters
             )

    ###CAREFULL!: to keep doing later optimizations you need to use the flipped 
    ##parameters!!! The whole optimization is flipped so you must continue 
    ##with those
    pol = large_scale_fitting(panels, out_params['coeffs'], pol_deg)
    clean_coeffs = apex_utitls.fit_coeffs_large_scale_removal(panels, out_params['coeffs'], pol)


    np.savez(os.path.join(plot_path, "panels_params_flipped.npz",
            params = 
            raw_params = out_params,
            losses=losses,
            gamma= gamma,
            F_pred=F_pred,
            F_gold=F,
            large_scale_coeffs = pol.parameters
             )




