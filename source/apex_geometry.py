import numpy as np
import matplotlib.pyplot as plt
from astropy import units as apu
from astropy import constants as cte
from geometry import cassegrain_silhouettes, subreflector_cone
import ipdb
import jax
import jax.numpy as jnp

def perfect_paraboloid(rv, tv, f, dr=None, dt=None):
    """
        This function produce a perfect paraboloid but also gives all the 
        necessary expressions that are needed when deformining it. 

        To compute the deformed  S_r and S_phi I need:
                er, et, der_dr, der_dt, det_dt, dn_dr, dn_dt, rho, r
        For math gibberish look at notes.
    """
    xv = rv*np.cos(tv)
    yv = rv*np.sin(tv)
    zv = rv**2/(4*f)
    p0 = apu.Quantity((xv.flatten(), yv.flatten(), zv.flatten())).T
    rho = np.sqrt(1+rv**2/(4*f**2))

    nx = -rv*np.cos(tv)/(2*f)/rho
    ny = -rv*np.sin(tv)/(2*f)/rho
    nz = np.ones(rv.shape)/rho
    n = apu.Quantity((nx.flatten(), ny.flatten(), nz.flatten())).T

    if(dr is None):
        dr = rv[0,1]-rv[0,0]
    if(dt is None):
        dt = tv[1,0]-tv[0,0]
    ds = (rho*rv*dr*dt).flatten()
    ##ok up to here is just normal stuffs
    
    er_x = np.cos(tv)/rho
    er_y = np.sin(tv)/rho
    er_z = rv/(2*f)/rho
    
    er = apu.Quantity((er_x.flatten(), er_y.flatten(), er_z.flatten())).T

    et_x = -np.sin(tv)
    et_y = np.cos(tv)
    et_z = np.zeros(tv.shape)
    et = apu.Quantity((et_x.flatten(), et_y.flatten(), et_z.flatten())).T

    der_dr_x = -rv*np.cos(tv)/(4*f**2*rho**3)
    der_dr_y = -rv*np.sin(tv)/(4*f**2*rho**3)
    der_dr_z = 1./(2*f*rho)-rv**2/(8*f**3*rho**3)
    der_dr = apu.Quantity((der_dr_x.flatten(), der_dr_y.flatten(), der_dr_z.flatten())).T

    der_dt_x = -np.sin(tv)/rho
    der_dt_y = np.cos(tv)/rho
    der_dt_z = np.zeros(tv.shape)
    der_dt = apu.Quantity((der_dt_x.flatten(), der_dt_y.flatten(), der_dt_z.flatten())).T

    det_dt_x = -np.cos(tv)
    det_dt_y = -np.sin(tv)
    det_dt_z = np.zeros(tv.shape)
    det_dt = apu.Quantity((det_dt_x.flatten(), det_dt_y.flatten(), det_dt_z.flatten())).T

    dn_dr_x = -np.cos(tv)/(2*f*rho**3)
    dn_dr_y = -np.sin(tv)/(2*f*rho**3)
    dn_dr_z = -rv/(4*f**2*rho**3)
    dn_dr = apu.Quantity((dn_dr_x.flatten(), dn_dr_y.flatten(), dn_dr_z.flatten())).T

    dn_dt_x = rv*np.sin(tv)/(2*f*rho)
    dn_dt_y = -rv*np.cos(tv)/(2*f*rho)
    dn_dt_z = np.zeros(tv.shape)
    dn_dt = apu.Quantity((dn_dt_x.flatten(),dn_dt_y.flatten(),dn_dt_z.flatten())).T

    return p0, n, ds, er, et, der_dr, der_dt, det_dt, dn_dr, dn_dt


def get_apex_panels(pr_v, pt_v, f, d1, 
    R       = [0.375, 1.265, 1.820, 2.605, 3.220, 4.040, 4.780, 5.435, 6.000],   ##check the first one.. I guess the hole should be smaller
    N       = [   12,    12,    24,    24,    48,    48,    48,    48       ],
    blockage=True,
    legs_diameter = 0.05*apu.m,
    secondary_diameter=0.75/2*apu.m,
    sigma_t=0.002,
    sigma_r=0.002,
    batch_size=None
    ):
    R = np.array(R)*apu.m
    p0, n0,ds,er,et,der_dr, der_dt, det_dt, dn_dr, dn_dt = perfect_paraboloid(pr_v,pt_v,f)
    if(blockage):
        block = cassegrain_silhouettes(p0, 
                                       legs_diameter=legs_diameter,
                                       secondary_diameter=secondary_diameter,
                                       sigma_t=sigma_t,
                                       sigma_r=sigma_r)
    else:
        block = np.ones(p0.shape[0])
    
    r = np.sqrt(p0[:,0]**2+p0[:,1]**2)
    phi = np.arctan2(p0[:,1],p0[:,0]).to_value(apu.rad)

    panels = dict()
    panels_points = 0
    for i in range(len(R)-1):
        r_mask = np.bitwise_and(r>R[i], r<R[i+1])
        ring_angle = 2*np.pi/N[i]
        for n in range(N[i]):
            panel_name = "{:d}{:02d}".format(i+1,n+1)
            angle1 = np.pi/2-n*ring_angle
            angle1 = np.arctan2(np.sin(angle1), np.cos(angle1)) ##just to be sure that is range
            angle2 = np.pi/2-(n+1)*ring_angle
            angle2 = np.arctan2(np.sin(angle2), np.cos(angle2))
            if(angle1==-np.pi):
               angle1 = np.pi
            angle_mask = np.bitwise_and(phi>=angle2, phi<angle1)
            mask = np.bitwise_and(r_mask, angle_mask) 
            ##compute the panel center
            r_middle = (R[i+1]+R[i])/2
            ang_middle = (angle2+angle1)/2
            ang_middle = np.arctan2(np.sin(ang_middle), np.cos(ang_middle))
            p_center = apu.Quantity([r_middle*np.cos(ang_middle), 
                                     r_middle*np.sin(ang_middle),
                                     r_middle**2/(4*f)])
            ##NOTE: to play with jax I should take out the units
            ##I could make more constant values that are being used
            r_panel = np.sqrt(p0[mask,0]**2+p0[mask,1]**2)
            rho_panel = np.sqrt(1+r_panel**2/(4*f**2))
            cte_sr = rho_panel+np.sum((p0[mask,:]-p_center)*der_dr[mask,:], axis=-1)
            cte1_st = np.sum((p0[mask,:]-p_center)*der_dt[mask,:], axis=-1)
            cte2_st = np.sum((p0[mask,:]-p_center)*det_dt[mask,:], axis=-1)
            x_ = np.sum((p0[mask,:]-p_center)*er[mask,:], axis=-1)
            y_ = np.sum((p0[mask,:]-p_center)*et[mask,:], axis=-1)

            panels_points += np.sum(mask)
            panels[panel_name] = {
                    'p0'        : p0[mask,:].to_value(apu.m),
                    'n0'        : n0[mask,:].decompose().to_value(apu.one),
                    'ds0'       : ds[mask].to_value(apu.m**2),
                    's_0r'      : (er[mask,:]*rho_panel[:,None]).decompose().to_value(apu.one),
                    's_0t'      : (et[mask,:]*r_panel[:,None]).to_value(apu.m),
                    'dn_dr'     : dn_dr[mask,:].to_value(1/apu.m),
                    'dn_dt'     : dn_dt[mask,:].decompose().to_value(apu.one),
                    'x_'        : x_.to_value(apu.m),
                    'y_'        : y_.to_value(apu.m),
                    'cte_sr'    : cte_sr.decompose().to_value(apu.one),
                    'cte1_st'   : cte1_st.to_value(apu.m),
                    'cte2_st'   : cte2_st.to_value(apu.m),
                    'r'         : r_panel.to_value(apu.m),
                    'blockage'  : block[mask]
                    }
        if(batch_size is None):
            print("You set batch_size to None, be carefull since if the points does not fit\
                    the batches they will not enter to the integral!")
        else:
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

    return panels


def build_apex_model(pr_v, pt_v, sr_v, st_v, 
                     primary_focus, f_d,
                     ##cone secondary
                     a=2796.11742*apu.mm, e=1.105262,
                     rc=30*apu.mm, Q=0.4284*apu.mm, C=0.5504*apu.mm,
                     #primary panels
                     R       = [0.375, 1.265, 1.820, 2.605, 3.220, 4.040, 4.780, 5.435, 6.000],
                     N       = [   12,    12,    24,    24,    48,    48,    48,    48       ],
                     #blockage
                     blockage=True,
                     legs_diameter = 0.05*apu.m,
                     secondary_diameter=0.75/2*apu.m,
                     sigma_t=0.002,
                     sigma_r=0.002,
                     batch_size=None
        ):
    """
        returns:
        -panels: dictionary where the keys are the panels names. Each item is other
                 dict with the following items:
                    -p0:    perfect paraboloid positions for the panel
                    -n0:    normal vector of the perfect paraboloid surface
                    -ds0:   differential surface for perfect paraboloid
                    -s_0r:  tangential unit vector for perfect paraboloid associated with r
                    -s_0t:  tangential unit vector for perfect paraboloid associated with phi
                    -dn_dr: derivate of n wr r
                    -dn_dt: derivate of n wr phi
                    -x_:    local coordinates of the panel positions wr the panel center
                    -y_:    local coordinates of the panel positions wr the panel center
                    -cte_sr: cte value for Sr when deforming the panel
                    -cte1_st: cte value for St when deforming the panel
                    -cte2_st: cte value for St when deforming the panel
                    -r:       r in cylindrical for the positions of the panels
                    -blockage: the blockage of the secondary and supporting legs that is projected
                                over the primary
                each item of this dictionary can be used as input for the deform_panel function

        -s_surf, s_n, s_ds: 
        -B: feed position
        -s_focus:   imaginary point behind the subreflector where the light rays joins. 
                    Can be used if you want to use the cos(n,s) term (found that is not needed)
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
    
    panels = get_apex_panels(pr_v, pt_v, primary_focus, d1, R=R, N=N,
                             blockage=blockage, legs_diameter=legs_diameter,
                             sigma_t=sigma_t, sigma_r=sigma_r,
                             batch_size=batch_size
                             )
    L = z_vertex
    B = m*(primary_focus-L)-L       ##feed position
    s_focus = np.sqrt(a**2+b**2)+z0    ##this is the imaginary point where the reflected points came from 
    
    return (panels, [s_surf_pos, s_n, s_ds], -B, s_focus, z_vertex)


##ok this part should be pure jax
def apply_panel_deformation(panels, coeffs):
    ##TODO: check if this is differentiable!!!
    deforms = [deform_panel(panels[name], coeffs[name]) for name in panels.keys()]
    surface, normal,ds, rms_deform  = (jnp.concatenate(x, axis=0) for x in zip(*deforms))
    return surface, normal, ds, rms_deform


def generate_start_coeffs(random_key, panel_names, start_rms=1e-5, dtype=jnp.float32):
    coeffs_out = dict()
    for name in panel_names:
        random_key, subkey = jax.random.split(random_key)
        #coeffs = jax.random.normal(subkey, shape=5)*start_rms
        coeffs = jax.random.uniform(subkey, shape=5, 
                                    minval=-start_rms, 
                                    maxval=start_rms)
        coeffs_out[name] = coeffs.astype(jnp.float32)
    return coeffs_out


def deform_function(x,y, coeffs):
    out = (coeffs[0]+coeffs[1]*x+coeffs[2]*y+
        coeffs[3]*(x**2+y**2)+coeffs[4]*(x**2-y**2))
    df_dx = coeffs[1]+2*(coeffs[3]+coeffs[4])*x
    df_dy = coeffs[2]+2*(coeffs[3]-coeffs[4])*y
    #If im not mistaken out should be in mts, df_dx and df_dy dimmensionless..
    return out, df_dx, df_dy


def deform_panel(panel_info, deform_coeffs):
    """
    Deform a posiitons.
    panel_info is one of the panel items returned by the get_apex_panels. It is 
    a dictionary with the following keys:
        p0      : ideal positions of the panel from the paraboloid eq
        n0      : ideal normal vector
        ds0     : ideal differential surface   
        s_0r
        s_0t
        dn_dr   : derivate of the normal vector w/r r
        dn_dt   : derivate of the normal vector w/r phi
        x_      : x in local coordinates of the panel (p0-p_center)\cdot e_r
        y_      : y in local coordinates of the panel (p0-p_cneter)\cdot e_t
        cte_sr: cte value for Sr when deforming the panel
        cte1_st: cte value for St when deforming the panel
        cte2_st: cte value for St when deforming the panel
        r:       r in cylindrical for the positions of the panels
        blockage: the blockage of the secondary and supporting legs that is projected
                    over the primary
    deform_coeffs: These are the deformation coefficients. Should be a list with
                    5 items.
    """
    deforms, df_dx, df_dy = deform_function(panel_info['x_'], panel_info['y_'], deform_coeffs)
    rms_deform = jnp.mean(deforms**2).reshape((-1,1))
    p = panel_info['p0']+deforms[:,None]*panel_info['n0']
    s_r = panel_info['s_0r']+(df_dx*panel_info['cte_sr'])[:,None]*panel_info['n0']+deforms[:,None]*panel_info['dn_dr']
    s_t = panel_info['s_0t']+(df_dx*panel_info['cte1_st']+
                              df_dy*(panel_info['cte2_st']+panel_info['r']))[:,None]*panel_info['n0']+\
                            +deforms[:,None]*panel_info['dn_dt']
                              
    normal = jnp.cross(s_r, s_t)
    normalization = jnp.sqrt(jnp.sum(normal**2, axis=-1))
    normal = normal/normalization[:,None]
    ds = normalization*panel_info['blockage']   ##still needs to be multiplied by dr and dphi
                                                ##the blockage itself should be at the E_i field
                                                ##but the 
    return p, normal, ds, rms_deform
