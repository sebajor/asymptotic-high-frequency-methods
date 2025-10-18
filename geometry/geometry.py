import jax
import jax.numpy as jnp
from astropy import units as apu
from astropy import constants as cte


"""
When having positions all the arrays are [:,3] where the second axis is the (x,y,z) 
coordinate positions
"""


def paraboloid_cylindrical(rv, tv, focus, diameter):
    """
    Since the ideal paraboloid has cylindrical symmetry then its best to parametrize
    it with a meshgrid of radius and angles
    rv, tv: meshgrid of the radius and the angle parameters. It should contain the
            proper units.
    focus:  paraboloid focus, in length units
    diameter: diameter of the paraboloid, in length units

    In cylindrical the parametrization is:
    R(r,theta) = (rcos(theat), rsin(theta), r/2f)

    This code returns the surface positions in mts,
    the normal vectors and the differental surface in mts**2
    """
    rv = rv.to_value(apu.m)
    
    xv = rv*np.cos(tv)
    yv = rv*np.sin(tv)
    zv = rv**2/(4*focus)
    
    surf_pos = jnp.array([xv,yv,zv]).T
    ##The tangent vectors are then: dR/dr =     (cos(theta), sin(theta), 1/2f)
    ##                              dR/dtheta = (-rsin(theta), rcose(theta), 0)
    ##And to compute the normal vector we make dR/dr x dR/dtheta 
    ## N = (-r**2/2/f cos(theta), -r**2/2/f sin(theta), r)
    nx = -rv**2/2/focus*jnp.cos(tv)
    ny = -rv**2/2/focus*jnp.sin(tv)
    nz = rv
    norm = np.array((nx, ny, nz))
    norm = norm/jnp.sqrt(jnp.sum(norm**2, axis=0))
    ##this one is just magnitude of the norm vector..
    ds = rv*jnp.sqrt(1+(rv/(2*focus))**2)*((rv[0,1]-rv[0,0])*(tv[1,0]-tv[0,0]))
    ds = ds.flatten()
    return surf_pos, norm, ds


