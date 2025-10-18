import numpy as np
import matplotlib.pyplot as plt
from astropy import units as apu
from astropy import constants as cte
import multiprocessing
import jax
import jax.numpy as jnp
from functools import partial
from jax import lax


###
### The kirchhoff fresnel integral is an improvement over the geometrical optics
### since its equivalent to the Hyugenss-Fresnel interpretation.. where when given
### an aperture each point in the aperture generates spherical waves and the 
### field value at the point P will be the sum of all the waves
###
### E(P) = -i*k/(4*pi)*integral(E(r')exp(i*k*(r+s))/(r*s)*(cos(n,s)-cos(n,r))dS
###
### Where 
###     r       :   is the distance from the source to the aperture
###     s       :   is the distnace from the aperture to point P
###     cos(n,s):   is the cos angle between the unit vectors n and s--> ie dot(n,s)/|s|
###     cos(n,r):   cos angle between n and r 
### 
### The standard formalism assume a that the source are localized.. since in principle
### we can compute the fields at the aperture we can reduce the equation to:
###
### E(r) = -i*k/(4*pi)*integral(E(r')exp(i*k*|r-r'|)/|r-r'| *cos(n,r) dS    --> Hyuggens-Fresnel
###
### This method does not compute the currents over the surface, ie the reflections 
### are geometrical-like but they are waves are spherical and interfer between them
### This also implies that the results are always scalar only.



def kirchhoff_propagation(surface_points, surface_normal, ds, incident_E, field_positions,
                          wavelength):
    k = 2*np.pi/wavelength
    E_out = np.zeros(field_positions.shape[0], dtype=complex)*apu.V/apu.m
    R = np.zeros(surface_points.shape)*apu.m
    for i in range(field_positions.shape[0]):
        R[:,0] = field_positions[i,0]-surface_points[:,0]
        R[:,1] = field_positions[i,1]-surface_points[:,1]
        R[:,2] = field_positions[i,2]-surface_points[:,2]
        
        r = np.sqrt(np.sum(R**2, axis=1))
        R_hat = R/r[:,None]
        
        cos_nr = np.sum(surface_normal*R_hat, axis=1)   ##dot product, the values should be normalized!
        ##dont know why xiadong takes abs of this...
        aux = np.exp(-1j*k*r)/r*incident_E*cos_nr*ds
        E_out[i] = 1j/(wavelength)*np.sum(aux)
    return E_out


def kirchhoff_propagation_vector(surface_points, surface_normal, ds, incident_E, field_positions,
                                 wavelength):
    """
    This one uses numpy vectorization.. but as it generates multidimensional matrices needs a lot of
    memory..
    """
    k = 2*np.pi/wavelength
    R = field_positions[None,:,:]- surface_points[:,None,:]
    r = np.sqrt(np.sum(R**2, axis=2))
    R /= r[:,:,None]
    cos_nr = np.sum(surface_normal[:,None,:]*R, axis=-1)
    E_r = 1j/(wavelength)*np.sum(np.exp(-1j*k*r)/r*incident_E[:,None]*cos_nr*ds[:,None], axis=0)
    return E_r




def kf_worker(surface_points, surface_normal, ds, incident_E, propagation_vector, 
              field_positions, wavelength, k, indices, E_r, shape):
    """
        surface_points      : (:,3) array with the postions of the surface
        surface_normal      : (:,3) normal vector of the surface
        ds                  : (:) differential area
        incident_E          : (:) incident field values
        propagation_vector  : (:,3) associated propagtion direction of E
        field_positions     : (:,3) position where you want to compute the field
        wavelenght          :
        k                   : k value
        indices             : indices fo the field_positions where the thread works
        E_r                 : reflected field
        shape               : shape of the reflected field
    """
    E_r_local = np.frombuffer(E_r, dtype=np.complex128).reshape(shape)
    
    for i in indices:
        rx = field_positions[i,0]-surface_points[:,0]
        ry = field_positions[i,1]-surface_points[:,1]
        rz = field_positions[i,2]-surface_points[:,2]
        R = apu.Quantity([rx, ry, rz]).T
        r = np.sqrt(np.sum(R**2, axis=1))
        R_hat = R/r[:,None]
        
        cos_nr = np.sum(surface_normal*R_hat, axis=1)   ##dot product, the values should be normalized!
        cos_sr = np.sum(surface_normal*propagation_vector, axis=1)
        cos = cos_nr+cos_sr
        aux = np.exp(-1j*k*r)/r*incident_E*cos*ds
        ##write data out
        E_r_local[i] = (1j/(2*wavelength)*np.sum(aux)).to_value(apu.V/apu.m)


def kf_worker_vector(surface_points, surface_normal, ds, incident_E, propagation_vector,
                     field_positions, wavelength, k, indices, E_r, shape):
    E_r_local = np.frombuffer(E_r, dtype=np.complex128).reshape(shape)
    R = field_positions[None,indices,:]- surface_points[:,None,:]
    r = np.sqrt(np.sum(R**2, axis=2))
    R /= r[:,:,None]
    cos_nr = np.sum(surface_normal[:,None,:]*R, axis=-1)
    E_r_local[indices] = (1j/(wavelength)*np.sum(np.exp(-1j*k*r)/r*incident_E[:,None]*cos_nr*ds[:,None], axis=0)).to_value(apu.V/apu.m)



def kirchhoff_propagation_batch(surface_points, surface_normal, ds, incident_E, propagation_vector,
                                field_positions, wavelength, batch_size=32, max_threads=6, vector_worker=False):
    if(vector_worker):
        func = kf_worker_vector
    else:
        func = kf_worker
    k = 2*np.pi/wavelength
    shape = field_positions.shape[0]
    arr_shape = np.prod(shape)*2    
    E_r = multiprocessing.Array('d', int(arr_shape), lock=False)

    batches = field_positions.shape[0]//batch_size
    remains = field_positions.shape[0]%batch_size

    lock = 1
    workers = []
    i = 0

    while(i< batches):
        if(len(workers)< max_threads):
            indices = np.arange(batch_size)+batch_size*i
            proc = multiprocessing.Process(target=func,
                                           args=(surface_points, 
                                                 surface_normal,
                                                 ds, 
                                                 incident_E,
                                                 propagation_vector,
                                                 field_positions,
                                                 wavelength,
                                                 k,
                                                 indices,
                                                 E_r,
                                                 shape)
                                               )
            workers.append(proc)
            proc.start()
            i+=1
        else:
            for j in range(len(workers)):
                proc = workers.pop()
                try:
                    proc.join()
                except:
                    print("Error joining thread!")
    ##
    print("joining last threads")
    for j in range(len(workers)):
        proc = workers.pop()
        try:
            proc.join()
        except:
            print("Error joining thread!")
    print("All threads joined")
    if(remains !=0):
        print("running the remained part")
        indices = np.arange(remains)+batch_size*i
        func(surface_points, surface_normal, ds,incident_E, propagation_vector, 
             field_positions,wavelength, k,indices, E_r, shape)
        ##finally we re-interpret once again the arrays
    #E_r = np.frombuffer(E_r.get_obj(), dtype=np.complex128).reshape(shape)
    #H_r = np.frombuffer(H_r.get_obj(), dtype=np.complex128).reshape(shape)
    E_r = np.frombuffer(E_r, dtype=np.complex128).reshape(shape)*apu.V/apu.m
    
    ##I need to compute the reflected propagation vector.
    return E_r


##These are the jax versions.. since after playing with the codes we realize that
## the cos_sr is not affecting the results we just omit it here..


@jax.jit
def kirchhoff_fresnel_single(surface_points, surface_normal, ds, incident_E, field_pos, wavel):
    """
    This function is for a single field_pos ie its a 3-size array
    """
    k = 2 * jnp.pi / wavel
    R = field_pos - surface_points  # [N_surface, 3]
    r = jnp.linalg.norm(R, axis=1)
    R_hat = R / r[:, None]
    cos_nr = jnp.sum(surface_normal * R_hat, axis=-1)
    E_r = (1j /(2*wavel)) * jnp.sum(jnp.exp(-1j * k * r) / r * incident_E * cos_nr * ds)
    return E_r


@partial(jax.jit,static_argnames=('chunk_size',))
def kirchhoff_fresnel_scan(surface_points, surface_normal, ds, incident_E,
                           field_positions, wavel, chunk_size=2048):
    """
    This implementation generates the batches with the lax.scan, and in each batch
    the GPU is being used with the vmaped version of the single point (ie each
    batch is solved in a vectorized way).
    """
    n_points = field_positions.shape[0]
    n_chunks = n_points//chunk_size
    kf_vmap = jax.vmap(kirchhoff_fresnel_single, in_axes=(None, None, None, None, 0, None))

    def body_fun(carry, idx):
        start = idx * chunk_size
        field_batch = lax.dynamic_slice(field_positions, (start, 0), (chunk_size, 3))

        # compute E_r for each point in this batch in parallel,
        #should I use block_until??
        E_batch = kf_vmap(surface_points, surface_normal, ds, incident_E, field_batch, wavel)
        carry = lax.dynamic_update_slice(carry, E_batch, (start,))
        return carry, None

    result_init = jnp.zeros(n_points, dtype=jnp.complex128)
    result, _ = lax.scan(body_fun, result_init, jnp.arange(n_chunks))
    return result



