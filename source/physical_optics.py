import numpy as np
from astropy import units as apu
from astropy import constants as cte
import multiprocessing


###
### The functions in this scripts use a incoming field, compute the induced currents
### over a PEC surface positions and then compute the projected fields in another position.
###
### Then, to use the functions you will need:
### 1) The points r on the surface where you want to compute the currents
### 2) The incoming fields in the points r
### 3) The normal vectors of the surface
### 4) The points where you want to compute the induced fields r'
### 5) dS = J*dx*dy on the surface, where J is the jacobian. We only need dS at r
###
### So you will need to have before hand, r,n,dS, H(r=r) and r' before hand
### and you will obtain E_r(r=r'), H_r(r=r') with the r subscript meaning the reflected field
###
### To make things easier we represent the 3D vectors as numpy arrays with size (..., 3)
### so x,y,z components are always in the second dimension of the arrays.
### 

def compute_induced_currents(surface_normal, incident_H):
    """
    surface_normal: normal vector at the points r where you want to compute the 
                    induced currents. np.array of size (..,3)
    incident_H:     Incident H field at the points r of the surface. 
                    np.array(..,3)*apu.A/apu.m
    ### in principle I think that there should be also a Jm that is induced by
    ### the incident_E...
    """
    Je = 2*np.cross(surface_normal, incident_H)
    return Je


def compute_reflected_fields(surface_points,ds, Je, field_positions, wavelength):
    """
    surface_points: r points where the induced current was computed. 
                    np.array(..,3)*apu.m
    ds:             differential surface element of the surface at the points r
                    np.array(...)*apu.m**2
    Je:             induced current at surface_points
    field_positions: the desired points r' where you want to compute the reflected
                    fields. np.array(..,3)*apu.m
    wavelength:     apu.m
    """

    k = 2*np.pi/wavelength
    E_r = np.zeros(field_positions.shape, dtype=complex)*apu.V/apu.m
    H_r = np.zeros(field_positions.shape, dtype=complex)*apu.A/apu.m

    R = np.zeros((surface_points.shape))*apu.m
    ###TODO: there should be a way to vectorize this and not need the loop
    for i in range(field_positions.shape[0]):
        ### we compute the vector distance between all the surface and one of the 
        ### field_points output
        R[:,0] = field_positions[i,0]-surface_points[:,0]
        R[:,1] = field_positions[i,1]-surface_points[:,1]
        R[:,2] = field_positions[i,2]-surface_points[:,2]

        r = np.sqrt(np.sum(R**2, axis=1))
        R_hat = R/r[:,None]
        ##
        phase = k*r
        r2 = k**2*r**2
        r3 = k**3*r**3

        ##This part is full of [:,None] becasue I need to match dimensions of the
        ##arrays.. some are scalars and then when operating them it complains...
        ##for what I read that is the best way to do it memory-wise
        aux0 = Je*(-1j/phase[:, None]-1/r2[:, None]+1j/r3[:,None])
        aux1 = (np.sum(Je*R_hat, axis=1)[:,None]*R_hat)*(1j/phase[:,None]+3/r2[:,None]-3*1j/r3[:,None])
        ee = (np.exp(-1j*phase)[:,None]*k**2)*(aux0+aux1)*ds[:,None]
        E_r[i,:] = np.sum(ee, axis=0)/(4*np.pi)*np.sqrt(cte.mu0/cte.eps0)
        ###
        aux0 = np.cross(Je, R_hat)
        aux1 = (1+1j*phase)/r2
        he = (np.exp(-1j*phase)*k**2)[:,None]*aux0*aux1[:,None]*ds[:,None]
        H_r[i,:]=np.sum(he, axis=0)/(4*np.pi)# *np.sqrt(cte.eps0/cte.mu0) ##for what I got I dont need to multiply for the 1/eta0
    return E_r,H_r



def reflected_field_worker(surface_points, ds, Je, field_pos, k, indices,
                           lock, E_r, H_r, shape):
    """
        indices is a list that iterates the field pos
    """
    ##I need to find a way to pass all by reference...
    ##these are when mp.Array lock=True
    #E_r_local = np.frombuffer(E_r.get_obj(), dtype=np.complex128).reshape(shape)
    #H_r_local = np.frombuffer(H_r.get_obj(), dtype=np.complex128).reshape(shape)
    E_r_local = np.frombuffer(E_r, dtype=np.complex128).reshape(shape)
    H_r_local = np.frombuffer(H_r, dtype=np.complex128).reshape(shape)
    for i in indices:
        rx = field_pos[i,0]-surface_points[:,0]
        ry = field_pos[i,1]-surface_points[:,1]
        rz = field_pos[i,2]-surface_points[:,2]
        R = apu.Quantity([rx, ry, rz]).T

        r = np.sqrt(np.sum(R**2, axis=1))
        R_hat = R/r[:,None]
        ##
        phase = k*r
        r2 = k**2*r**2
        r3 = k**3*r**3

        aux_e0 = Je*(-1j/phase[:, None]-1/r2[:, None]+1j/r3[:,None])
        aux_e1 = (np.sum(Je*R_hat, axis=1)[:,None]*R_hat)*(1j/phase[:,None]+3/r2[:,None]-3*1j/r3[:,None])
        ee = (np.exp(-1j*phase)[:,None]*k**2)*(aux_e0+aux_e1)*ds[:,None]
        ###
        aux_h0 = np.cross(Je, R_hat)
        aux_h1 = (1+1j*phase)/r2
        he = (np.exp(-1j*phase)*k**2)[:,None]*aux_h0*aux_h1[:,None]*ds[:,None]
        ##
        ## here we write the data out
        #lock.acquire()
        E_r_local[i,:] = (np.sum(ee, axis=0)/(4*np.pi)*np.sqrt(cte.mu0/cte.eps0)).to_value(apu.V/apu.m)
        H_r_local[i,:]= (np.sum(he, axis=0)/(4*np.pi)).to_value(apu.A/apu.m)
        #lock.release()


def compute_reflected_fields_batch(surface_points,ds, Je, field_positions, wavelength,
                                  batch_size=32, max_threads=6):
    k = 2*np.pi/wavelength
    shape = field_positions.shape
    arr_shape = np.prod(shape)*2    
    #E_r = np.zeros(field_positions.shape, dtype=complex)*apu.V/apu.m
    #H_r = np.zeros(field_positions.shape, dtype=complex)*apu.A/apu.m
    E_r = multiprocessing.Array('d', int(arr_shape), lock=False)
    H_r = multiprocessing.Array('d', int(arr_shape), lock=False)

    ##also I need to create shared objects for surface_points, ds, Je, and field_pos
    ##otherwise the data is being copy at each thread.. for that I need to use shared_memory
    
    
    batches = field_positions.shape[0]//batch_size
    remains = field_positions.shape[0]%batch_size

    #lock = multiprocessing.Lock()
    lock = 1
    workers = []
    i = 0
    while(i< batches):
        if(len(workers)< max_threads):
            indices = np.arange(batch_size)+batch_size*i
            proc = multiprocessing.Process(target=reflected_field_worker,
                                           args=(surface_points, 
                                                 ds,
                                                 Je,
                                                 field_positions,
                                                 k,
                                                 indices,
                                                 lock,
                                                 E_r,
                                                 H_r,
                                                 shape
                                                 )
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
        reflected_field_worker(surface_points, ds, Je, field_positions,k, indices,
                               lock, E_r, H_r, shape)
        ##finally we re-interpret once again the arrays
    #E_r = np.frombuffer(E_r.get_obj(), dtype=np.complex128).reshape(shape)
    #H_r = np.frombuffer(H_r.get_obj(), dtype=np.complex128).reshape(shape)
    E_r = np.frombuffer(E_r, dtype=np.complex128).reshape(shape)*apu.V/apu.m
    H_r = np.frombuffer(H_r, dtype=np.complex128).reshape(shape)*apu.A/apu.m
    return E_r, H_r

