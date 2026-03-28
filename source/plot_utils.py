import numpy as np
import matplotlib.pyplot as plt
from apex_geometry import deform_function
import matplotlib.cm as cm
import matplotlib.tri as tri
from astropy.modeling import models, fitting


def plot_deformations(panels, coeffs, ax=None, correct_global=True, pol_deg=2,
                      vmin=-50*1e-6, vmax=50*1e-6):
    """
    """
    if(ax is None):
        flag = 1
        fig, ax = plt.subplots(1,1)
    else:
        flag = 0

    if(correct_global):
        x_data = np.array([])
        y_data = np.array([])
        z_data = np.array([])
        for name in panels.keys():
            x_data = np.concatenate([x_data, panels[name]['p0'][:,0]])
            y_data = np.concatenate([y_data, panels[name]['p0'][:,1]])
            deforms, df_dx, df_dy = deform_function(panels[name]['x_'], panels[name]['y_'], coeffs[name])
            z_data = np.concatenate([z_data, deforms])
        p_init = models.Polynomial2D(pol_deg)
        fit_p = fitting.LevMarLSQFitter()
        p = fit_p(p_init, x_data, y_data, z_data) 
    else:
        p = lambda x,y: 0


    for name in panels.keys():
        if(name == 'fake'):
            continue
        x = panels[name]['p0'][:,0]
        y = panels[name]['p0'][:,1]
        triang = tri.Triangulation(x, y)
        deforms, df_dx, df_dy = deform_function(panels[name]['x_'], panels[name]['y_'], coeffs[name])
        ax.tripcolor(triang, deforms-p(x,y), vmin=vmin, vmax=vmax, cmap=cm.jet)
    if(flag):
        return fig, ax
    else:
        return 

