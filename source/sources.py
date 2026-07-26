import numpy as np
from astropy import units as apu
from astropy import constants as cte
import jax.numpy as jnp
#import ipdb


class plane_wave_source():
    ### 
    ###
    def __init__(self, 
                 freq, 
                 k_hat=np.array([0,0,1]), 
                 E0=np.array([1,0,0])*apu.V/apu.m, 
                 x0=np.zeros(3)*apu.m):
        """
        k_hat: kx,ky,kz  --> this should be normalized
        x0 = x0,y0,z0
        E0 = Ex, Ey, Ez

        The simplest is to take:
            k_vector = (0,0,1)
            x0 = (0,0,0)
            E0 = (E,0,0)
        btw, you have to ensure the orthogonality between E,H,k
        """
        wavel = cte.c/freq
        self.k_hat = k_hat
        self.k = 2*np.pi/wavel*k_hat
        self.r = x0
        self.E0 = E0

    def propagate(self, 
                  positions, 
                  eta = np.sqrt(cte.mu0/cte.eps0)
                  ):
        ###The positions (...,3)
        #ipdb.set_trace()
        R = positions-self.r[None, :]
        E = np.ones(positions.shape, dtype=complex)*self.E0[None,:]
        E = E*np.exp(-1j*np.sum(self.k*R, axis=1))[:,None]
        H = np.cross(self.k_hat[None, :], E)/eta
        return E, H
        


class cylindrical_gaussian_beam():
    """
    edge tapper: in dB
    wavel:
    origin:     where the source is located
    direction:  propagation direction
    local_x:    if you want to force x' vector (as its cyclindrical symmetry it doesnt affect)
    
    I take the standard definition with the propagation in z and w0 at z=0
    this are the local coordinates of the gaussian source, then when propagting
    to the actual system positions I transform them to the local coords.
    """

    def __init__(self,edge_tapper, horn_aperture, wavel, origin, direction, local_y=None):
        self.w0 = horn_aperture/np.sqrt(np.abs(10*np.log(10**(edge_tapper.to_value(apu.dB)/20))))
        self.z_c = np.pi*self.w0**2/wavel
        self.wavel = wavel
        self.origin = origin
       
        self.direction = direction
        x,y,z = self.generate_local_coords(direction, ref=local_y)
        self.direction_vectors = np.column_stack((x,y,z))
    
    def generate_local_coords(self, direction, ref=None):
        #from the k we compute the x' and y' at the orthogonal plane
        ##first get a reference vector to create one of the axis
        if(ref is None):
            ref = np.array((1,0,0)) 
            if(np.sum(ref*direction)>0.9):
                ref = np.array((0,1,0))
        x = np.cross(ref, direction)
        x = x/np.sum(x**2)
        y = np.cross(direction,x)
        y = y/np.sum(y**2)
        return x,y,direction
        

    def syscoord2local(self, positions):
        """
            convert system coordinates to the local ones
            postions: array (..,3)
        """
        new_pos = positions-self.origin[None,:]
        local_pos = np.matmul(self.direction_vectors,new_pos.T)  ##matrix product
        return local_pos.T

    def propagate(self, positions):
        """
        compute beam at a given system position.
        positions: array(..,3) 
        """
        local_pos = self.syscoord2local(positions)
        k = 2*np.pi/self.wavel
        w = self.w0*np.sqrt(1+(local_pos[:,2]/self.z_c)**2)
        R = local_pos[:,2]+self.z_c**2/local_pos[:,2]
        phi = np.arctan2(self.z_c, local_pos[:,2]).to_value(apu.rad)
        r2 = local_pos[:,0]**2+local_pos[:,1]**2
        field = np.sqrt(2/(np.pi*w**2))*np.exp(-r2/w**2 -
                                               1j*k*local_pos[:,2]-
                                               1j*np.pi*r2/(self.wavel*R)+
                                               1j*phi)
        field = field*apu.V ##to have units of V/m
        return field


        
def propagate_cylindrical_gaussian_beam(
        edge_tapper, horn_aperture, wavel, origin, target_pos
        ):
    """
    It does the same as the class, but in a single function to be jax friendly....
    Also to avoid overhead it always points towards z
    edge_tapper: should be in dB
    horn_aperture: in mts
    wavel:          in mts
    origin:
    target_positions:   jnp.array with the positions where you want to calculate the field
    """
    w0 = horn_aperture/jnp.sqrt(jnp.abs(10*jnp.log(10**(edge_tapper/20))))
    z_c = jnp.pi*w0**2/wavel
    local_pos = target_pos-origin[None, :]
    k = 2*jnp.pi/wavel
    w = w0*jnp.sqrt(1+(local_pos[:,2]/z_c)**2)
    R = local_pos[:,2]+z_c**2/local_pos[:,2]
    phi = jnp.arctan2(z_c, local_pos[:,2])
    r2 = local_pos[:,0]**2+local_pos[:,1]**2
    field = jnp.sqrt(2/(jnp.pi*w**2))*jnp.exp(-r2/w**2 -
                                           1j*k*local_pos[:,2]-
                                           1j*jnp.pi*r2/(wavel*R)+
                                           1j*phi)
    return field



def propagate_cylindrical_gaussian_beam_offset(
        edge_tapper, horn_aperture, offset, rotation, wavel, origin, target_pos
        ):
    """
    It does the same as the class, but in a single function to be jax friendly....
    This one gives an offset to the origin and also rotate the gaussian beam axis
    edge_tapper: should be in dB
    horn_aperture: in mts
    offset:         (x,y,z)
    rotations:      in rad
    wavel:          in mts
    origin:
    target_positions:   jnp.array with the positions where you want to calculate the field
    """
    w0 = horn_aperture/jnp.sqrt(jnp.abs(10*jnp.log(10**(edge_tapper/20))))
    z_c = jnp.pi*w0**2/wavel
    origin = pos_origin+offset
    r_global = target_pos-origin
    alpha = rotation[0]; beta = rotation[1]; gamma = rotation[2]
    R = jnp.array([[jnp.cos(alpha)*jnp.cos(beta), jnp.cos(alpha)*jnp.sin(beta)*jnp.sin(gamma)-jnp.sin(alpha)*jnp.cos(gamma), jnp.cos(alpha)*jnp.sin(beta)*jnp.cos(gamma)+jnp.sin(alpha)*jnp.sin(gamma)],
                   [jnp.sin(alpha)*jnp.cos(beta), jnp.sin(alpha)*jnp.sin(beta)*jnp.sin(gamma)+jnp.cos(alpha)*jnp.cos(gamma), jnp.sin(alpha)*jnp.sin(beta)*jnp.cos(gamma)-jnp.cos(alpha)*jnp.sin(gamma)],
                   [-jnp.sin(beta)              , jnp.cos(beta)*jnp.sin(gamma)                                             , jnp.cos(beta)*jnp.cos(gamma)                                             ]]
                  )
    ##local coordinates
    local_pos = jnp.matmul(R, orig_sec.T).T
    k = 2*jnp.pi/wavel
    w = w0*jnp.sqrt(1+(local_pos[:,2]/z_c)**2)
    R = local_pos[:,2]+z_c**2/local_pos[:,2]
    phi = jnp.arctan2(z_c, local_pos[:,2])
    r2 = local_pos[:,0]**2+local_pos[:,1]**2
    field = jnp.sqrt(2/(jnp.pi*w**2))*jnp.exp(-r2/w**2 -
                                           1j*k*local_pos[:,2]-
                                           1j*jnp.pi*r2/(wavel*R)+
                                           1j*phi)
    return field

