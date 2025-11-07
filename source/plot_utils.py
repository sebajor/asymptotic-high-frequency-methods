import numpy as np
import matplotlib.pyplot as plt
from apex_geometry import deform_function
import matplotlib.cm as cm
import matplotlib.tri as tri


def plot_deformations(panels, coeffs, ax=None, vmin=-50*1e-6, vmax=50*1e-6):
    if(ax is None):
        flag = 1
        fig, ax = plt.subplots(1,1)
    else:
        flag = 0

    for name in panels.keys():
        triang = tri.Triangulation(panels[name]['p0'][:,0], panels[name]['p0'][:,1])
        deforms, df_dx, df_dy = deform_function(panels[name]['x_'], panels[name]['y_'], coeffs[name])
        ax.tripcolor(triang, deforms, vmin=vmin, vmax=vmax, cmap=cm.jet)
    if(flag):
        return fig, ax
    else:
        return 

