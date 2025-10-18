import numpy as np
import matplotlib.pyplot as plt
from astropy import units as apu
from astropy import constants as cte
import ipdb



def deformed_plane(xv,yv, params):
    """
    Computes a deformed plane with the following deformations
    zv = a+bx+cy+d(x-y)+e(x+y)
    that is equivalent a change in the piston, tilt and a rotational deformation
    
    this function also return the normal vector of the plane and the surface element

    xv, yv: meshgrid with units of length
    params: [a,b,c,d,e] or [piston, x tilt, y tilt, deform 1, deform 2]
    """
    x = xv.to_value(apu.m)
    y = yv.to_value(apu.m)
    z = (params[0]+
         params[1]*x+
         params[2]*y+
         params[3]*(x**2+y**2)+
         params[4]*(x**2-y**2))*apu.m
    ##normal vector nx = dz/dx, ny=dz/dy
    n_x = params[1]+2*x*(params[3]+params[4])   
    n_y = params[2]+2*y*(params[3]-params[4])
    norm = np.array((-n_x, -n_y, np.ones(n_x.shape)))
    norm = norm/np.sqrt(np.sum(norm**2, axis=0))
    #norm = norm.reshape((3,-1)).T           ##CHECK!!!
    norm = norm.reshape((3,-1))
    norm = norm.swapaxes(0,1)
    ds = np.sqrt(1+n_x**2+n_y**2)               ##Jacobian
    ds = ds*(xv[0,1]-xv[0,0])*(yv[1,0]-yv[0,0]) ##to have units of m**2

    #plane_positions = apu.Quantity([xv.flatten(), yv.flatten(), z.flatten()]).T
    plane_positions = apu.Quantity([xv.flatten(), yv.flatten(), z.flatten()])
    plane_positions = plane_positions.swapaxes(0,1)
    ds =  ds.flatten()
    return plane_positions, norm, ds


def deformed_circular_reflector(xv, yv, r, params):
    plane_positions, norm, ds = deformed_plane(xv,yv,params)
    mask = (plane_positions[:,0]**2+plane_positions[:,1]**2)< r**2
    return plane_positions[mask,:], norm[mask,:], ds[mask]




def paraboloid_cartesian(xv, yv, focus, diameter):
    """
        z = x**2+y**2/(4f)
        The normal are toward the inside of the paraboloid
        ##the parametrization is not good, it looks weird
    """
    z = (xv**2+yv**2)/(4*focus)
    mask = xv**2+yv**2<(diameter/2)**2
    ##compute normal vectors
    nx = xv/2/focus
    ny = yv/2/focus
    norm = np.array((-nx, -ny, np.ones(nx.shape)))
    norm = norm/np.sum(norm**2, axis=0)
    #norm = norm.reshape((3,-1)).T           ##CHECK!!!
    norm = norm.reshape((3,-1))
    norm = norm.swapaxes(0,1)

    ds = np.sqrt(1+nx**2+ny**2)
    ds = ds*(xv[0,1]-xv[0,0])*(yv[1,0]-yv[0,0]) ##to have units of m**2
    surf_pos = apu.Quantity((xv.flatten(), yv.flatten(), z.flatten()))
    surf_pos = surf_pos.swapaxes(0,1)
    return surf_pos, norm, ds, mask


def paraboloid_cylindrical(rv, tv, focus, diameter): 
    #in cilindrical the parametrization is:
    ## R(r,theta) = (rcos(theat), rsin(theta), r/2f)
    ###The diameter here does nothing.. I should give that at the rv
    xv = rv*np.cos(tv)
    yv = rv*np.sin(tv)
    zv = rv**2/(4*focus)
    surf_pos = apu.Quantity([xv.flatten(),yv.flatten(),zv.flatten()]).T
    ##compute normal vectors
    ##The tangent vectors are then: dR/dr =     (cos(theta), sin(theta), 1/2f)
    ##                              dR/dtheta = (-rsin(theta), rcose(theta), 0)
    ##And to compute the normal vector we make dR/dr x dR/dtheta 
    ## N = (-r**2/2/f cos(theta), -r**2/2/f sin(theta), r)
    nx = -rv**2/2/focus*np.cos(tv)
    ny = -rv**2/2/focus*np.sin(tv)
    nz = rv
    norm = np.array((nx, ny, nz))
    norm = norm/np.sqrt(np.sum(norm**2, axis=0))
    norm = norm.reshape((3,-1)).T           ##CHECK!!!
    #norm = norm.reshape((3,-1))
    #norm = norm.swapaxes(0,1)

    #norm = apu.Quantity([nx.flatten(),ny.flatten(),nz.flatten()]).T
    #norm = norm/np.sqrt(np.sum(norm, axis=1))
    ##this one is just magnitude of the norm vector..
    ds = rv*np.sqrt(1+(rv/(2*focus))**2)*(rv[0,1]-rv[0,0])*(tv[1,0]-tv[0,0])
    ds = ds.flatten()
    return surf_pos, norm, ds



def hyperboloid_cylindrical(rv, tv, a,b):
    ### the hyperboloid points satisfy
    ### z(r) =  +-a*sqrt(r**2/b**2-1)
    ###
    xv = rv*np.cos(tv)
    yv = rv*np.sin(tv)
    zv = a*np.sqrt(1+rv**2/b**2)-a
    surf_pos = apu.Quantity([xv.flatten(),yv.flatten(),zv.flatten()]).T
    dz_dr = a*rv/(b**2*np.sqrt(rv**2/b**2+1))
    nx = -dz_dr*np.cos(tv)/(np.sqrt(1+dz_dr**2))
    ny = -dz_dr*np.sin(tv)/(np.sqrt(1+dz_dr**2))
    nz = 1/(np.sqrt(1+dz_dr**2))
    norm = np.array((nx,ny,nz))
    norm = norm/np.sqrt(np.sum(norm**2, axis=0))
    norm = norm.reshape((3,-1)).T           ##CHECK!!!
    ds = rv*np.sqrt(1+dz_dr**2)*(rv[0,1]-rv[0,0])*(tv[1,0]-tv[0,0])
    ds = ds.flatten()
    return surf_pos, norm, ds


def cassegrain_cylindrical(pr_v, pt_v, sr_v, st_v,
                           primary_focus,  f_d,
                           s =1
                           ):
    """
        pr_v: primary radii values (meshgrid)
        pt_v: primaey theta values (meshgrid)
        sr_v: secondary radii values (meshgrid)
        st_v: secondary theta values (meshgrid)
        f_d:  cassegrain sys focal ratio (f/D) 

        s: oversize of the secondary, this controls the spillover (s>=1)

        B: is the focal plane position.. in principle it will be at (0,0,B)
    """
    ##NOTE I AM NOT SURE AT ALL OF THE RELATIONS!!!!!! MAYBE I AM MISSING SOMETHING!!
    d1 = np.max(pr_v)*2
    d2 = np.max(sr_v)*2
    F_eff = f_d*d1
    m = F_eff/primary_focus         ##magnification
    L = primary_focus*(1-d2/d1/s)   ##vertex position of the hyperboloid
    B = m*(primary_focus-L)-L       ##feed position
    ###secondary parameters
    z0 = (primary_focus-B)/2
    c = (primary_focus+B)/2
    a = L-z0
    b = np.sqrt(c**2-a**2)
    s_focus = np.sqrt(a**2+b**2)+z0    ##this is the imaginary point where the reflected points came from 
    ###
    s_surf_pos, s_n, s_ds = hyperboloid_cylindrical(sr_v, st_v, a,b)
    s_surf_pos[:,2] += L  #z0   ##shouldnt be z0+a (?)
    p_surf_pos, p_n, p_ds = paraboloid_cylindrical(pr_v, pt_v, primary_focus, d1)
    return ([p_surf_pos, p_n, p_ds], [s_surf_pos, s_n, s_ds], -B, s_focus)




def subreflector_cone(rv, tv, a=2796.11742*apu.mm, e=1.105262, 
                      rc=30*apu.mm, Q=0.4284*apu.mm, C=0.5504*apu.mm):
    """
    Generates the alma subreflector
    rv>rc --> sqrt(a**2+(r**2/(e**2-1)))-a
    rv<rc --> sqrt(a**2+(r**2/(e**2-1)))-a- (Q*((rc-r)/rc)**2+C*((rc-r)/rc)**3)
    """
    b = a*np.sqrt(e**2-1)
    xv = rv*np.cos(tv)
    yv = rv*np.sin(tv)
    zv = np.zeros(rv.shape)*apu.m
    nx = np.zeros(rv.shape)
    ny = np.zeros(rv.shape)
    nz = np.zeros(rv.shape)
    j = np.zeros(rv.shape)*apu.m

    ##this is the standard hyperboloid
    mask = rv>rc
    r_local = rv[mask]
    t_local = tv[mask]
    zv[mask]= a*np.sqrt(1+r_local**2/b**2)-a

    dz_dr = (a*r_local/(b**2*np.sqrt(r_local**2/b**2+1))).decompose()
    nx[mask] = (-dz_dr*np.cos(t_local)/(np.sqrt(1+dz_dr**2))).decompose()
    ny[mask] = (-dz_dr*np.sin(t_local)/(np.sqrt(1+dz_dr**2))).decompose()
    nz[mask] = (1/(np.sqrt(1+dz_dr**2))).decompose()
    j[mask] = (r_local*np.sqrt(dz_dr**2+1)).decompose()

    ##here the hyperboloid become a cone
    mask = rv<rc
    r_local = rv[mask]
    t_local = tv[mask]
    aux = (rc-r_local)/rc
    zv[mask] = np.sqrt(a**2+(r_local**2/(e**2-1)))-a-Q*aux**2-C*aux**3

    dz_dr = (a*r_local/(b**2*np.sqrt(r_local**2/b**2+1))-2*Q/rc*aux+3*C/rc*aux**2).decompose()
    nx[mask] = (-dz_dr*np.cos(t_local)/(np.sqrt(dz_dr**2+1))).decompose()
    ny[mask] = (-dz_dr*np.sin(t_local)/(np.sqrt(dz_dr**2+1))).decompose()
    nz[mask] = (1/(np.sqrt(1+dz_dr**2))).decompose()
    j[mask] = (r_local*np.sqrt(dz_dr**2+1)).decompose()
    ##

    surf_pos = apu.Quantity([xv.flatten(),yv.flatten(),zv.flatten()]).T
    norm = np.array((nx, ny,nz))
    norm = norm/np.sqrt(np.sum(norm**2, axis=0))
    norm = norm.reshape((3,-1)).T           ##CHECK!!!
    ds = j*(rv[0,1]-rv[0,0])*(tv[1,0]-tv[0,0])
    ds = ds.flatten()
    return surf_pos, norm, ds



def cassegrain_cylindrical_cone(pr_v, pt_v, sr_v, st_v,
                           primary_focus,  f_d, 
                           a=2796.11742*apu.mm, e=1.105262, 
                           rc=30*apu.mm, Q=0.4284*apu.mm, C=0.5504*apu.mm
        ):
    """
    The usual cylindrical cassegrain, but as subreflector use the modified 
    hyperboloid with a cone in the vertex
        pr_v: primary radii values (meshgrid)
        pt_v: primaey theta values (meshgrid)
        sr_v: secondary radii values (meshgrid)
        st_v: secondary theta values (meshgrid)
        f_d:  cassegrain sys focal ratio (f/D) 
        a:    a parameter of the secondary
        e:    eccentricity of the secondary
        rc:   radius where the cone starts
        Q:    2nd order cone parameter
        C:    3rd order cone parameter
    """
    d1 = np.max(pr_v)*2
    d2 = np.max(sr_v)*2
    F_eff = f_d*d1
    m = F_eff/primary_focus         ##magnification
    c = e*a
    b = a*np.sqrt(e**2-1)
    z0 = primary_focus-c            ##position of the center of the hyperbola
    z_vertex = z0+a
    
    s_surf_pos, s_n, s_ds = subreflector_cone(sr_v, st_v, a=a, e=e,
                      rc=rc, Q=Q, C=C)
    s_surf_pos[:,2] += z_vertex
    p_surf_pos, p_n, p_ds = paraboloid_cylindrical(pr_v, pt_v, primary_focus, d1)

    L = z_vertex
    B = m*(primary_focus-L)-L       ##feed position
    s_focus = np.sqrt(a**2+b**2)+z0    ##this is the imaginary point where the reflected points came from 

    return ([p_surf_pos, p_n, p_ds], [s_surf_pos, s_n, s_ds], -B, s_focus)

    


