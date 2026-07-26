import numpy as np
import matplotlib.pyplot as plt
from astropy import units as apu
from astropy import constants as cte
import astropy
import os, yaml
import ipdb
from scipy.optimize import minimize
import subprocess
import datetime
import shutil


def Aperture_to_Surface(E,xv, yv, wavel, f_prim, debug=False):
    """
    E       : aperture
    r2      : radius square meshgrid (xv**2+yv**2) in area units
    wavel   : wavelength in distance units
    f_prim  : primary focal length in distance units
    """
    if(debug):
        print("debug Aperture_to_Surface")
        step = xv[0,1]-xv[0,0]; N = xv.shape[0]
        X = np.linspace(-N/2, N/2, N)
        xv, yv = np.meshgrid(X,X)*step
    r2 = xv**2+yv**2
    Z = np.angle(E)*wavel/(4*np.pi)
    S = -Z*np.sqrt(1+r2/(4*f_prim**2))
    #S = wavel/(4*np.pi)*np.angle(E)*np.sqrt(1+r2/(4*f_prim**2))
    return S

def Surface_to_Aperture(S, r2, wavel, f_prim):
    """
    This only returns the phase of the aperture..
    r2: radius square meshgrid (xv**2+yv**2) in area units
    """
    phi = 4*np.pi/wavel*S/np.sqrt(1+r2/(4*f_prim**2))
    return np.exp(-1j*phi)


def Normalize_surface_data(data, mask):
    total = np.sum(data[mask])
    n = np.sum(mask)
    out = data-total/n
    return out

def calculate_surface_rms(data, mask):
    amplitude = Normalize_surface_data(data, mask)
    rms = np.sum((data[mask].to_value(apu.um))**2)
    n = np.sum(mask)
    out = (rms/n)**0.5
    return out


def cassegrain_blockage(R1=6*apu.m, r2=0.75/2*apu.m, legs=0.3*apu.m):
    """
    """
    def block_func(x,y):
        r = np.sqrt(x**2+y**2)
        mask = np.zeros(r.shape, dtype=bool)
        mask[r<R1] = True
        mask[np.abs(x)<legs] = False
        mask[np.abs(y)<legs] = False
        mask[r<r2] = False
        return mask
    return block_func


####Functions only related to apex
def get_holo1_data(config, account, scan, date=None):
    """
    Just call the script in the observation account to transform it to an holo1
    file and transfer it to paniri.
    This just made an ssh connection and ask for the holo1 file.. the actual code
    is at the observational account
    """
    if(date is None):
        date = datetime.datetime.now()
        date = date.strftime("%Y-%m-%d")
    filename = "APEX-"+scan+"-"+date+"-T-0109.F-9995A-2022"
    cmd = 'ssh '+account+' \'bash -lc "./PackAndShip.csh '+filename+'"\''
    #ret = subprocess.call(cmd.split(" "), shell=True)
    ret = os.system(cmd)
    print(ret)
    ##TODO: return the location of the .holo file!
    return ret


def regrid_data(config, holo1_file_path, map_type="lores"):
    """
    map_type: can be lores or hires
    """
    cmd = (config['regrid_parameters']['regrid_exec']+" --filename "+ 
           os.path.expanduser(holo1_file_path)+
           " --map_size "+str(config['regrid_parameters'][map_type]['arcsec_size'])+
           " --fwhm "+str(config['regrid_parameters']['fwhm'])+
           " --kernel_rad "+str(config['regrid_parameters']['kernel_radius'])+
           " --max_threads"+str(config['regrid_parameters']['threads'])+ 
           " --new_samples "+ str(config['regrid_parameters'][map_type]['samples']))
    ret = os.system(cmd)
    name = os.path.basename(holo1_file_path).split('.holo')[0]
    os.makedirs(os.path.join(config['regrid_parameters'][map_type]['file_dir'], name), exist_ok=True)
    dest = os.path.join(config['regrid_parameters'][map_type]['file_dir'], name)
    scan_path = os.path.join(dest, name+'.reg')
    shutil.move(os.path.expanduser(os.path.join(os.path.curdir,"output.reg")), scan_path)
    print("Saving regrid data to {:}".format(scan_path))
    return scan_path




def read_regrid_data(filename):
    u,v,amp,phase= np.loadtxt(filename).T
    N = int(np.sqrt(len(amp)))
    F = (amp*np.exp(1j*phase)).reshape((N,N))
    u = u.reshape((N,N))*apu.deg
    v = v.reshape((N,N))*apu.deg
    ##the grid is done starting from the most positive number and move backwards
    ##(these was done to be comparable with the old regrid).. Then it causes
    ##problems when plotting.
    #u = u[:,::-1]
    #v = v[::-1,:]
    u = -u
    v = -v
    #F = F[::-1,::-1]
    return F,u,v



def read_regrid_parra(config, rg_folder, map_type='hires'):
    """
    This the old regrid output that divides the amp and phase in different files,
    it doesnt saves the position of the regrided points.
    Also has a dirty trick that generates a denoise in the aperture 
    Read rgamp and rgpha from a given folder.
    The rg_folder can be a path or just a scan number stored at the default location
    in the configuration file
    map_type: hires o lores
    """
    ##check if its a valid path
    if(os.path.exists(rg_folder)):
        ##check if its an absolute path that is not default one necessarilly
        if(rg_folder.endswith('/')):
           rg_folder = rg_folder[:-1]
        path = os.path.join(rg_folder, os.path.basename(rg_folder))
    else:
        ##check if its at the default location
        path = os.path.join(config['regrid_parameters'][map_type]['file_dir'], rg_folder, rg_folder)
    amp = np.loadtxt(path+'.rgamp')
    phase = np.loadtxt(path+'.rgpha')
    N = int(np.sqrt(len(amp)))
    F = (amp*np.exp(1j*phase)).reshape((N,N))    ##check phase sign!
    #F = F.T   ###CHECK!!!!
    #F = np.swapaxes(F,0,1)
    #F = F[::-1,:]       ##check!!!
    map_size = config['regrid_parameters'][map_type]['arcsec_size']/3600.
    u = np.linspace(-map_size/2, map_size/2, F.shape[0])*apu.deg
    v = np.linspace(-map_size/2, map_size/2, F.shape[0])*apu.deg
    return F, u, v



def dirty_trick_denoise(F,uv, vv, percentage=0.98, mask_data=False,
                        r_mask=0.125*apu.deg, l_mask=0.001*apu.deg):
    """
    trick to denoise the aperture map. For this we renormalize the amplitude data
    using the minimum value as reference. This minimum value its far away of the 
    main beam, then it is a high frequency component of the aperture and then 
    lowering its value acts as a high freq filter.
    One big problem is that you mess up the scale of the map then the maps are not
    comparable
    This trick is done since the HP vector voltmeter dont have enough range
    mask_center: 
    mask_legs: 
    """
    amp = np.abs(F)
    phase = np.angle(F)
    if(mask_data is False):
        amp -= np.min(amp)*percentage
    else:
        mask = np.ones(F.shape, dtype=bool)
        r = np.sqrt(uv**2+vv**2)
        mask[r<r_mask] = False
        mask[np.abs(uv)<l_mask] = False
        mask[np.abs(vv)<l_mask] = False
        amp[mask] -= np.min(amp[mask])*percentage
    out = amp*np.exp(1j*phase)
    return out
    
    

def phase_correction(F,uv,vv,wavel,d1=2.18*apu.m,d2=7.485*apu.m):
    """
    Phase compensation. This function used to be in the regrid code but for easy
    reading was moved here. We are not 100% sure about it, but we guess that 
    cames from the pathlength difference between the main and the reference 
    receivers... Also we dont know what actually are the distances d1 and d2.
    """
    tetha = np.sqrt(uv**2+vv**2)
    correction = 2*np.pi/wavel*(d1-d2)*(1-np.cos(tetha.to_value(apu.rad)))*apu.rad
    phase = np.angle(F)*apu.rad
    phase += correction
    amp = np.abs(F)
    out = amp*np.exp(1j*phase.to_value(apu.rad))
    return out




def APEX_panel_area(panelid, x, y, dr=0*apu.m, dtheta=0*apu.rad):
    """
    panelid:
    x,y: meshgrid with length units
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
    return out, ring_mask


def APEX_mask(x,y, panels=[537,538], rings=[1,8], r1=6*apu.m, r2=0.75/2*apu.m, 
              legs=0.3*apu.m, dr=0):
    """
    x,y: meshgrid
    """
    block = cassegrain_blockage(R1=r1, r2=r2, legs=legs)
    cassegrain_mask = block(x,y)
    panel_mask = np.zeros(cassegrain_mask.shape, dtype=bool)
    for panel in panels:
        p_mask, r_mask = APEX_panel_area(panel,x,y,dr=dr)
        panel_mask = np.bitwise_or(panel_mask, p_mask)
    ring_mask = np.zeros(cassegrain_mask.shape, dtype=bool)
    for ring in rings:
        panel = ring*100+1
        p_mask, r_mask = APEX_panel_area(panel,x,y,dr=dr)
        ring_mask = np.bitwise_or(ring_mask, r_mask)
    mask = np.bitwise_and(np.bitwise_and(cassegrain_mask, 
                                       np.invert(panel_mask)),
                                        np.invert(ring_mask))
    return mask, cassegrain_mask, panel_mask, ring_mask



