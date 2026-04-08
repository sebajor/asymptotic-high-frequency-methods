from apex_geometry import *
from scipy import optimize
from scipy import interpolate
from astropy import units as apu

###
### This code is to make a bridge with the old apex holography pipeline based 
### in the nearfield FFT approach in Baars that returns a surface error after
### several corrections.
### This code make a fitting for each panel zone to get the parameters in the 
### format that the jax implementation likes.. 
### The previous implementation doesnt care about the parameters, 
### just want the height value at the actuator positions and then it doesnt uses
### the panel local coordinates that the current implementation has.
###



def plane_deform(x, params):
    xv, yv = x
    z = (params[0]+params[1]*xv+params[2]*yv +
        params[3]*(xv**2+yv**2)+params[4]*(xv**2-yv**2))
    ###this deforamtion is weird, the units dont match...
    return z

def plane_fit(params, data, xv, yv, fit_func=plane_deform):
    xv = xv.to_value(apu.m)
    yv = yv.to_value(apu.m)
    data = data.to_value(apu.um)
    plane_model = fit_func((xv,yv), params)
    error = np.abs(plane_model-data)
    return error


def APEX_panel_area(panelid, x, y,
                    panel_x, panel_y,
                    dr=0*apu.m, dtheta=0*apu.rad):
    """
    panelid:
    x,y: meshgrid with length units

    panel_x, panel_y: these comes from the pytree
    """
    #           0       1      2      3      4      5      6      7      8
    R       = [0.375, 1.265, 1.820, 2.605, 3.220, 4.040, 4.780, 5.435, 6.000]
    N       = [   12,    12,    24,    24,    48,    48,    48,    48       ]

    R = [ x*apu.m for x in R]

    ring = panelid//100
    panel = panelid%100

    r = np.sqrt(x**2+y**2)
    angle = np.arctan2(y,x)

    
    ring_mask = np.bitwise_and(r<(R[ring]-dr), r>(R[ring-1]+dr))
    
    panels_angle = 2*np.pi/N[ring-1]
    ##against all normal conventions.. the panel 0 is at pi/2 and the indexing is 
    ##clockwise >:(
    angle1 = np.pi/2-(panel-1)*panels_angle
    ##we need to put the angles in the same representation
    angle1 = np.arctan2(np.sin(angle1),np.cos(angle1))*apu.rad
    angle2 = np.pi/2-(panel)*panels_angle
    angle2 = np.arctan2(np.sin(angle2),np.cos(angle2))*apu.rad
    if(angle1.value==-np.pi):
        angle1 = np.pi*apu.rad
    angle1 -= dtheta.to(apu.rad)
    angle2 += dtheta.to(apu.rad)
    angle_mask = np.bitwise_and(angle>=angle2, angle<angle1)
    out = np.bitwise_and(angle_mask, ring_mask)

    ## these are the same as the previous but with the pytree
    r = np.sqrt(panel_x**2+panel_y**2)
    angle = np.arctan2(panel_y, panel_x)
    ring_mask = np.bitwise_and(r<(R[ring]-dr), r>(R[ring-1]+dr))
    angle_mask = np.bitwise_and(angle>=angle2, angle<angle1)
    out_pytree = np.bitwise_and(angle_mask, ring_mask)

    return out, ring_mask, out_pytree





def compatiblity_panel_fit(surface_error, xv, yv, panels, fit_func=plane_deform,
                            dr=8*apu.cm, dtheta=12./64*np.pi*apu.rad
                           ):
    """
    surface_error: surface error in apu.um
    xv, yv: surface error rectangular meshgrid in apu.m
    panels: jax pytree defined in apex_geometry script
    fit_func: 
    dr:     area to left out from the panel
    dtheta: area to left out from the panel 

    Since the jax implenetation uses cylindrical parametrization we need to get the
    panel area and the interpolate the missing samples before doing anything
    """
    coeffs = dict()
    x = xv[0,:].to_value(apu.m)
    y = yv[:,0].to_value(apu.m)
    interp = interpolate.RegularGridInterpolator((x,y), surface_error.to_value(apu.um))

    for panel_name, panel in panels.items():
        p_x, p_y, p_z = panel['p0'].T
        ring = int(panel/100)
        n = N[ring-1]
        p_mask, r_mask, pytree_mask  = APEX_panel_area(
                int(panel_name), xv, yv,
                p_x*apu.m, p_y*apu.m,
                dr=dr,
                dtheta=dtheta/n
                )
        x_fit = xv[p_mask].to_value(apu.m).flatten()
        y_fit = yv[p_mask].to_value(apu.m).flatten()
        z = surface_error[p_mask].to_value(apu.um).flatten()
        interp = interpolate.RegularGridInterpolator((x_fit, y_fit), z)
        ##Since the sampling wont match we need to interpolate the data
        p_x = p_x[pytree_mask]
        p_y = p_y[pytree_mask]
        p_z_interp = interp(np.array(p_x, p_y))
        ##ok, now we can do the fitting in the panels local coordinate
        x_ = panel['x_'][pytree_mask]
        y_ = panel['y_'][pytree_mask]
        params = np.zeros(5)
        res_lsq = optimize.least_squares(
                fun=plane_fit,
                x0=params,
                args=(
                    p_z_interp,
                    x_,
                    y_,
                    fit_func
                    ),
                method='trf',
                tr_solver='exact',
                loss='linear'
                )
        par = res_lsq.x
        coeffs[panel_name] = np.array(par)
    return coeffs
