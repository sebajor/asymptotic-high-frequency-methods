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


parser = argparse.ArgumentParser(
    description="Geometrical fit for apex hologrpahy pipeline")

parser.add_argument('-f', '--filename', dest='filename' , type=str,
                    help="Input filename should be a .reg or .hdf5 output of the fourier pipeline")
parser.add_argument('-lr', '--learn_rate', dest='learning_rate', type=float, default=1e-4, 
                    help="learning rate for the optimizer")
parser.add_argument('-mi', '--max_iters', dest='max_iters', type=int, default=1000,
                    help="maximum iterations of the optimization")
parser.add_argument('-cl', '--conv_lim', dest='conv_lim', type=float, default=4*1e-14,
                    help="convergence limit")
parser.add_argument("-ll", "--loss_lim", dest='loss_lim', type=float, default=3*1e-11,
                    help="lower limit of the loss")
parser.add_argument("-pi","--plot_interval",dest='plot_interval', type=int, default=10,
                   help="How often generate the debugging plots")
parser.add_argument("-plot_path", dest='plot_path', type=str, default="~/MODULES/physical_optics/")
parser.add_argument("-no_stop", "--no_stop", dest="no_stop", action='store_true',
                    help="Avoids all the stop mechanism, the optimization runs up to the max iteration")


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
    ampamp = np.abs(F)
    out = amp*np.exp(1j*phase.to_value(apu.rad))
    return out


plot_dir = os.path.abspath(os.path.expanduser(args.plot_path))
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
    F = np.conj(F)
    ##TODO: check that this flip is correct!
    ##F = F[::-1,::-1]
    plot_path = os.path.join(plot_dir, filename.split('.reg')[0])

elif(filename.endswith('.hdf5')):
    f = h5py.File(filepath, 'r')
    F = np.array(f['phase_correction']['data'])
    F = F/F[128,128]
    F = np.conjugate(F)
    f.close()
    plot_path = os.path.join(plot_dir, os.path.split(os.path.split(os.path.split(filepath)[0])[0])[1])

else:
    print("Unsuported input file.. must be .reg or .hdf5!")
    sys.exit(1)

print("Creating the plot directories at %s"%plot_path)
os.makedirs(plot_path, exist_ok=True)
plot_path = os.path.join(plot_path, 'geometry_fit')
os.makedirs(plot_path, exist_ok=True)

##computing the target aperture
E_shift = np.fft.ifftshift(F)
gold_aperture = np.fft.fftshift(np.fft.ifft2(E_shift))




learning_rate = args.learning_rate
gamma = 1                   ##here we dont care.. the deformation rms is zero always
iters = args.max_iters
loss_lim = args.loss_lim
conv_lim = args.conv_lim
plot_interval = args.plot_interval

map_dtype = jnp.complex128


###
###Here we start to define everything! Note that the hyperparameters are in hyperparameters file!
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
coeffs = generate_start_coeffs_zeros(panels.keys(), dtype=jnp.float32)
s_n = jnp.array(s_n).astype(jnp.float32)
s_ds = jnp.array(s_ds.to_value(apu.m**2)).astype(jnp.float32)

horn_position = (jnp.array((0,0,B.to_value(apu.m))).T).astype(jnp.float32)
edge_tapper= jnp.array(edge_tapper.to_value(apu.dB))
horn_aperture = jnp.array(horn_aperture.to_value(apu.m))


##create the forward function
def make_forward_function(coeffs,panels, s_pos0, s_n0, s_ds, sec_vertex,
                    target_pos, horn_position, 
                     wavel, batch_size, target_distance, map_dtype=jnp.complex128):
    @jax.jit
    def forward_function( 
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


forw_function = make_forward_function(coeffs, panels, s_pos, s_n, s_ds, sec_vertex.to_value(apu.m),
                     target_pos, horn_position, wavel.to_value(apu.m),
                     batch_size, target_distance.to_value(apu.m),
                     map_dtype=map_dtype)


#def loss_funct(coeffs, sec_rotation, sec_offsets, gold, gamma):
def loss_funct(params, gold, gamma, norm_point=256*128+128):
    pred, deform_mse = forw_function(
                                     params['edge_tapper'], params['horn_aperture'], 
                                     params['horn_offsets'], params['horn_rotation'],
                                     params['sec_offsets'], params['sec_rotation'])
    #pred_norm = pred/jnp.max(jnp.abs(pred))
    pred_norm = pred/pred[norm_point]
    #error = pred_norm-gold
    #
    #this one is in the aperture
    pred_reshape = pred_norm.reshape((256,256))
    pred_shift = jnp.fft.ifftshift(pred_reshape)
    aperture = jnp.fft.fftshift(jnp.fft.ifft2(pred_shift)).flatten()
    error = aperture-gold
    ##we use the same loss either way
    loss = jnp.mean(error.real**2+error.imag**2)+gamma*jnp.mean(deform_mse)
    #jax.debug.print("jax.debug.print(y) -> {y}", y=jnp.mean(deform_mse))
    return loss

params = {
        #"coeffs":coeffs,
        "sec_rotation": sec_rotation,
        "sec_offsets": sec_offsets,
        'edge_tapper': edge_tapper,
        'horn_aperture': horn_aperture,
        'horn_offsets': horn_offsets,
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


####
####
####
def plot_debug(params, gold_aperture, F, target_points, plot_path, iteration):
    fig, ax = plt.subplots(2,2,figsize=(10,10))
    pred, deform_mse = forw_function(
                                     params['edge_tapper'], params['horn_aperture'],
                                     params['horn_offsets'], params['horn_rotation'],
                                     params['sec_offsets'], params['sec_rotation'])
    F_pred = pred/pred[257*128]
    F_pred = np.array(F_pred).reshape((target_points, target_points))

    ax[0,0].plot(20*np.log10(np.abs(F_pred[:,128])), color='darkblue')
    ax[0,0].plot(20*np.log10(np.abs(F[:,128])), color='darkred')
    ax[0,0].set_ylim(-82, 2)

    ax[0,1].plot(20*np.log10(np.abs(np.diag(F_pred))), color='darkblue')
    ax[0,1].plot(20*np.log10(np.abs(np.diag(F))), color='darkred')
    ax[0,1].set_ylim(-82, 2)

    E_shift = np.fft.ifftshift(F_pred)
    aperture = np.fft.fftshift(np.fft.ifft2(E_shift))
    ax[1,0].imshow(np.angle(aperture))
    ax[1,1].imshow(np.abs(aperture))
    title = "iteration "+str(iteration)+"\n"
    title += "suref_pos:%.5f %.5f %.5f"%(params['sec_offsets'][0]*1e3,
                               params['sec_offsets'][1]*1e3,
                               params['sec_offsets'][2]*1e3)+"  "

    title += "subref_rotation: %.5f %.5f %.5f"%(
            np.rad2deg(params['sec_rotation'][0])*1e3,
            np.rad2deg(params['sec_rotation'][1])*1e3,
            np.rad2deg(params['sec_rotation'][2])*1e3)+"\n"
    title += "horn pos: %.5f %.5f %.5f"%(params['horn_offsets'][0]*1e3,
                                         params['horn_offsets'][1]*1e3,
                                         params['horn_offsets'][2]*1e3
            )+"   "
    title += "horn rotation: %.5f %.5f %.5f"%(np.rad2deg(params['horn_rotation'][0])*1e3,
                                         np.rad2deg(params['horn_rotation'][1])*1e3,
                                         np.rad2deg(params['horn_rotation'][2])*1e3
            )+"\n"
    title += "edge tapper %.4f horn aperture %.4f"%(params['edge_tapper'], params['horn_aperture'])
    #ax.set_title("iteration "+str(i))
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
    print("Starting Geometrical fitting")
    for i in range(args.max_iters):
        start = time.time()
        params, opt_state, loss = train_step(params, opt_state, gold_aperture.flatten(),gamma)
        losses.append(loss)
        log_msg = "iter:%i \t loss:%E \t time:%.3f "%(i, loss, time.time()-start)
        logging.info(log_msg)
        if(i%plot_interval==0):
            plot_debug(params, gold_aperture, F, target_points, plot_path, i)
        if(not args.no_stop):
            if(np.mean(losses[-10:]) < loss_lim):
                print("Loss limit reached at iteration %i: %E < %E"%(i, loss, loss_lim))
                break
            if(np.mean(np.abs(np.diff(losses[-10:])))< conv_lim):
                print("Loss convergence reached at iteration %i: %E < %E"%(i, np.mean(np.abs(np.diff(losses[-10:])))), conv_lim)
                break

    print("Out of the optimization loop")
    out_params = pytree_to_numpy(params)
    pred, deform_mse = forw_function(
                                     params['edge_tapper'], params['horn_aperture'],
                                     params['horn_offsets'], params['horn_rotation'],
                                     params['sec_offsets'], params['sec_rotation'])
    F_pred = pred/pred[257*128]
    F_pred = np.array(F_pred).reshape((target_points, target_points))

    np.savez(os.path.join(plot_path, "geo_params.npz"),
                params=out_params,
                losses=losses,
                F_pred=F_pred,
                F_gold=F)

