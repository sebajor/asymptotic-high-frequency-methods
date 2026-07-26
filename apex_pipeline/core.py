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


###enable the 64bit computation
os.system("export JAX_ENABLE_X64=True")
jax.config.update('jax_enable_x64', True)


##optimization parameters
learning_rate = 1e-6#1e-3
gamma = 5*1e7

##TODO: here we need to pre-process the incomming beam map, ie do the regrid
##and substract the global features.




###
###


##we can play with the subreflector position+rotations
##also there can be problems with receiver position+tilts
##finally we also could have problems with the transmitter position and distance.
##
##In the computational side we can also have problems with the sampling..We are
##using an equidistant sampling which is not the best.

##We are not 100% sure about correctness of these values!
sec_offsets = [0*apu.mm, 0*apu.mm, 15*apu.mm]       #default holo defocus
sec_rotations = [0*apu.mdeg, 0*apu.mdeg, 0*apu.mdeg] #default holo rotations


panels, s_pos, s_n, s_ds, sec_vertex, B, target_pos = create_apex_geometries(r_min_prim, d1, r_points,
                      r_min_sec, d2, 
                      t_points_primary, t_points_secondary,
                      f1, f_d, 
                      silhouette, legs_diameter, secondary_silhouette,
                      sigma_t, sigma_r,
                      target_distance, target_map_size, target_points,
                      batch_size=batch_size
                      )


