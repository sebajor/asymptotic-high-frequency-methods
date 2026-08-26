import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting


def flip_panels_coeffs(coeffs_flipped):
    """
    Because how the resampling is done we got that the beam map is flipped 
    in the x and y axis... We cannot flip it with F = F[::-1, ::-1] since it mess
    up the sampling (eg the main peak is not at 128,128 anymore) adding a tilt
    in the aperture... For consistency with the Fourier pipeline we cant modify the
    sampling, therefore we just exchange the panels coefficients at the end..
    For example the panel 101 info is contained in the panel 107.
    Note that if you later want to compute the beam once again you will need to 
    use the flipped coefficients, since the geometric paramaters are also computed
    over that flipped data >:(
    """
    coeffs_real = dict()
    N       = [   12,    12,    24,    24,    48,    48,    48,    48       ],
    for p in range(len(N)):
        for n in range(N[p]//2):
            p_top = str((p+1)*100+n+1)
            p_bottom = str((p+1)*100+n+1+N[p]//2)
            coeffs_real[p_top] = coeffs_flipped[p_bottom]
            coeffs_real[p_bottom] = coeffs_flipped[p_top]
    return coeffs_real

def large_scale_fitting(panels, coeffs, pol_deg=4):
    """
    Make a global fitting over the inferred data
    """
    x_data = []
    y_data = []
    z_data = []
    for name in panels.keys():
        if(name == 'fake'):
            continue
        x_data = np.concatenate([x_data, panels[name]['p0'][:,0]])
        y_data = np.concatenate([y_data, panels[name]['p0'][:,1]])
        deforms, df_dx, df_dy = deform_function(panels[name]['x_'], panels[name]['y_'], coeffs[name])
        z_data = np.concatenate([z_data, deforms])
    p_init = models.Polynomial2D(pol_deg)
    fit_p = fitting.LevMarLSQFitter()
    p = fit_p(p_init, x_data, y_data, z_data)
    return p




###
### Ploting codes 
###



def APEX_get_panel_nodes(
    rings = [0.375, 1.265, 1.820, 2.605, 3.220, 4.040, 4.780, 5.435, 6.000],
    ring_name = [  'A',   'B',   'C',   'D',   'E',   'F',   'G',   'H'  ],
    n_panels = [   12,    12,    24,    24,    48,    48,    48,    48  ]
        ):
    """
    Calculate the initial/ending radius/angle for each panel
    """
    r_init = []
    r_end = []
    angle_init = []
    angle_end = []
    v_number = []
    v_id = []
    names = []

    for i in range(8):
        panels = n_panels[i]
        theta = np.linspace(0, 2*np.pi, panels+1)
        r0 = rings[i]*np.ones(panels)
        theta0 = theta[:-1]
        r1 = rings[i+1]*np.ones(panels)
        theta1 = theta[1:]
        ##
        vnumber = np.flipud( np.roll(np.arange(panels)+1 ,-panels//4))  ##This is bcs the panel numbering is the worst
        vid = 100*(i+1)+vnumber
        name = [ring_name[i]+str(x) for x in range(1,panels+1)]
        r_init += r0.tolist()
        r_end += r1.tolist()
        angle_init += theta0.tolist()
        angle_end += theta1.tolist()
        v_number += vnumber.tolist()
        v_id += vid.tolist()
        names += name

    return names, r_init, r_end, angle_init, angle_end, v_number, v_id

def APEX_draw_panels(ax, alpha=0.5, show_names=False,
        rings = [0.375, 1.265, 1.820, 2.605, 3.220, 4.040, 4.780, 5.435, 6.000],
        ring_name = [  'A',   'B',   'C',   'D',   'E',   'F',   'G',   'H'  ],
        n_panels = [   12,    12,    24,    24,    48,    48,    48,    48  ]
        ):
    names, r_i, r_e, a_i, a_e, vn, vi =  APEX_get_panel_nodes(rings=rings, ring_name=ring_name,
                                                         n_panels=n_panels)

    ##first we need to draw the lines in r
    x_r = [r_i*np.cos(a_i), r_e*np.cos(a_i)]
    y_r = [r_i*np.sin(a_i), r_e*np.sin(a_i)]

    for x,y in zip(x_r,y_r):
        ax.plot(x_r, y_r, alpha=alpha, lw=0.5, c=[0.6,0.6,0.6])

    if(show_names):
        xlabel = (np.array(r_e)-0.1)*np.cos((np.array(a_i)+np.array(a_e))/2)
        ylabel = (np.array(r_e)-0.1)*np.sin((np.array(a_i)+np.array(a_e))/2)
        rotation = np.arctan2(ylabel, xlabel)*180/np.pi-90
        for x,y,panel_id, rot in zip(xlabel, ylabel, vi, rotation):
            ax.text(x,y,panel_id, fontsize=6, color=(0.8,0.8,0.8,1),
                    ha='center',va='center', rotation=rot)
    ##now we need to join the radius
    radii = np.unique(np.append(r_i, rings[-1]))
    theta = np.linspace(-np.pi, np.pi, 361)
    for r in radii:
        ax.plot(r*np.cos(theta), r*np.sin(theta), alpha=alpha, lw=0.5,
                color=(0.6,0.6,0.6))
    ##support legs
    ct = np.cos(theta)
    st = np.sin(theta)
    cc = (0.6,0.6,0.6)

    ax.fill( 0.1*ct-4.6,0.05*st,color=cc,lw=0.5)
    ax.fill( 0.1*ct-4.3,0.05*st,color=cc,lw=0.5)
    ax.fill( 0.1*ct-3.0,0.05*st,color=cc,lw=0.5)
    ax.fill( 0.1*ct-5.8,0.05*st,color=cc,lw=0.5)

    ax.fill( 0.1*ct+4.6,0.05*st,color=cc,lw=0.5)
    ax.fill( 0.1*ct+4.3,0.05*st,color=cc,lw=0.5)
    ax.fill( 0.1*ct+3.0,0.05*st,color=cc,lw=0.5)
    ax.fill( 0.1*ct+5.8,0.05*st,color=cc,lw=0.5)

    ax.fill( 0.05*ct,0.1*st+4.6,color=cc,lw=0.5)
    ax.fill( 0.05*ct,0.1*st+4.3,color=cc,lw=0.5)
    ax.fill( 0.05*ct,0.1*st+3.0,color=cc,lw=0.5)
    ax.fill( 0.05*ct,0.1*st+5.8,color=cc,lw=0.5)

    ax.fill( 0.05*ct,0.1*st-4.6,color=cc,lw=0.5)
    ax.fill( 0.05*ct,0.1*st-4.3,color=cc,lw=0.5)
    ax.fill( 0.05*ct,0.1*st-3.0,color=cc,lw=0.5)
    ax.fill( 0.05*ct,0.1*st-5.8,color=cc,lw=0.5)


def get_APEX_actuator_positions(screw_file='cfg/screw_table.csv', rd=0.1, td=0.025):
    data = np.loadtxt(screw_file, delimiter=',', skiprows=1)
    panel_number = data[:,0].astype(int)
    r = data[:,1::2]
    angle = data[:,2::2]
    x = r*np.cos(angle)
    y = r*np.sin(angle)
    ##calculate the text posoitions
    x1 = (r[:,0]+rd)*np.cos(angle[:,0]-td)
    y1 = (r[:,0]+rd)*np.sin(angle[:,0]-td)
    x2 = (r[:,1]+rd)*np.cos(angle[:,1]+td)#-td)
    y2 = (r[:,1]+rd)*np.sin(angle[:,1]+td)#-td)
    x3 = (r[:,2]-rd)*np.cos(angle[:,2]-td)
    y3 = (r[:,2]-rd)*np.sin(angle[:,2]-td)
    x4 = (r[:,3]-rd)*np.cos(angle[:,3]+td)
    y4 = (r[:,3]-rd)*np.sin(angle[:,3]+td)

    x_text = np.vstack((x1,x2,x3,x4,x[:,4])).T
    y_text = np.vstack((y1,y2,y3,y4,y[:,4])).T
    return panel_number,x,y, x_text, y_text


def get_APEX_panel_center(panel, foci):
    R = [0.375, 1.265, 1.820, 2.605, 3.220, 4.040, 4.780, 5.435, 6.000]
    N = [   12,    12,    24,    24,    48,    48,    48,    48       ]
    
    r_panel = int(panel/100)
    n_panel = panel%100-1         ##panel num in the ring
    ring_angle = 2*np.pi/N[r_panel-1]

    angle1 = np.pi/2-n_panel*ring_angle
    angle1 = np.arctan2(np.sin(angle1), np.cos(angle1))
    angle2 = np.pi/2-(n_panel+1)*ring_angle
    angle2 = np.arctan2(np.sin(angle2), np.cos(angle2))
    if(angle1==-np.pi):
        angle1 = np.pi
    r_middle = (R[r_panel]+R[r_panel-1])/2
    ang_middle = (angle2+angle1)/2
    ang_middle = np.arctan2(np.sin(ang_middle), np.cos(ang_middle))
    p_center = np.array([r_middle*np.cos(ang_middle), 
                         r_middle*np.sin(ang_middle),
                         r_middle**2/(4*foci)]).T
    return p_center


def cartesian_to_panel_coord(x,y, panel_id, foci):
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)
    z = r**2/(4*foci)
    p0 = np.array([x,y,z]).T
    
    rho = np.sqrt(1+r**2/(4*foci**2))
    ##
    er_x = np.cos(theta)/rho
    er_y = np.sin(theta)/rho
    er_z = r/(2*foci)/rho
    er = np.array([er_x.flatten(), er_y.flatten(), er_z.flatten()]).T

    et_x = -np.sin(theta)
    et_y = np.cos(theta)
    et_z = np.zeros(theta.shape)
    et = np.array([et_x.flatten(), et_y.flatten(), et_z.flatten()]).T
    
    p_center = get_APEX_panel_center(panel_id, foci)
    
    x_ = np.sum((p0-p_center)*er, axis=-1)
    y_ = np.sum((p0-p_center)*et, axis=-1)
    return x_, y_


def deform_function(x,y, coeffs):
    out = (coeffs[0]+coeffs[1]*x+coeffs[2]*y+
        coeffs[3]*(x**2+y**2)+coeffs[4]*(x**2-y**2))
    df_dx = coeffs[1]+2*(coeffs[3]+coeffs[4])*x
    df_dy = coeffs[2]+2*(coeffs[3]-coeffs[4])*y
    #If im not mistaken out should be in mts, df_dx and df_dy dimmensionless..
    return out, df_dx, df_dy


#TODO:CHECK APEX_generate_screw_table, cartesian_to_panel_coord and get_APEX_panel_center!!!!!
###         THE OTHER IMPORTANT THING TO DO IS TO FIT THE GLOBAL POLYNOMIAL!!!!
def APEX_generate_screw_table(coeffs, large_scale_pol=lambda x,y: 0,
                              screw_file='cfg/screw_table.csv', limit_microns=10, screw_lead=36./25,
                              foci =4.8,text_size=3,
                              mask_rings = [], mask_panels=[],
                              **kwargs):
    """
    coeffs: coefficients generated by the optimization procedure. Its a pytree with the 
    large_scale_pol: Large scale polynomial to discount to the inferred surface.
                    astropy.models.Polynomial2D. Its a lambda function.

    screw_file: txt with the actuators positions 
    limit_microns: dont show adjument under this threshold
    screw_lead
    foci:   parabooid foci
    """
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set(xlabel='[m]', ylabel='[m]',title='Panel error ')
    APEX_draw_panels(ax, show_names=True)
    panel_num,y, x, y_text, x_text = get_APEX_actuator_positions(screw_file)
    for panel, act_x, act_y, xtext, ytext in zip(panel_num, x,y,x_text, y_text):
        ##check that the panel is not in the masked ones
        p_ring = int(panel/100)
        if(p_ring in mask_rings):
            continue
        elif(panel in mask_panels):
            continue
        ##We need to change to the local coordinate of the panel...
        x_, y_ = cartesian_to_panel_coord(act_x,act_y,panel, foci)
        panel_coeffs = coeffs[str(panel)]
        inferred_deform, _, _ = deform_function(x_, y_, panel_coeffs)    ##deform are in m!
        deform = np.array([z-large_scale_pol(a_x, a_y) for z,a_x,a_y in zip(inferred_deform, act_x, act_y)])
        turns = -deform*1e6*screw_lead
        for t, z, x_txt, y_txt in zip(turns,deform, xtext, ytext):
            if(abs(z*1e6)<limit_microns):
                continue
            color = 'r' if t>0 else 'b'
            ax.annotate("%+i"%t, (x_txt,y_txt), color=color,ha='center', va='center',fontsize=text_size, **kwargs)
    return fig, ax
            

