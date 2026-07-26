import numpy as np
import matplotlib.pyplot as plt
from astropy import units as apu
from astropy import constants as cte
import astropy
import os, yaml
import ipdb
from scipy.optimize import minimize
from holo_utils import *
import multiprocessing


def BeamPattern_to_Aperture(F, u, v, wavel, phase_zero=[2.5*apu.m, 2.5*apu.m],
                            distance=1835*apu.m, high_terms=False,
                            threaded=True, max_threads=8,
                            batch_size=256
                            ):
    """
    """
    if(not high_terms):
        return BeamPattern_to_Aperture_classical(F,u,v,wavel, phase_zero=phase_zero)

    if(threaded):
        uv, vv = np.meshgrid(u,v)
        high_order, fft_brute, xv, yv = fft_high_term_batch(uv,vv, 
                                                        F, wavel, R=distance,
                                                        max_threads=max_threads, 
                                                        batch_size=batch_size)
        x_shift = xv[0,:]
        y_shift = yv[:,0]
    else:
        uv,vv = np.meshgrid(u,v)
        k = 2*np.pi/wavel
        du = u[1]-u[0]; dv = v[1]-v[0]
        x,y = np.fft.fftfreq(len(u), du), np.fft.fftfreq(len(v), dv)
        if type(u) == apu.quantity.Quantity:
            if astropy.__version__ < '4':
                x_shift = np.fft.fftshift(x) * x.unit * wavel * apu.rad
                y_shift = np.fft.fftshift(y) * y.unit * wavel * apu.rad
            else:
                x_shift = np.fft.fftshift(x) * wavel * apu.rad
                y_shift = np.fft.fftshift(y) * wavel * apu.rad
        else:
            x_shift = np.fft.fftshift(x) * wavel.to_value(apu.m)
            y_shift = np.fft.fftshift(y) * wavel.to_value(apu.m)
        xv, yv = np.meshgrid(x_shift, y_shift)
        xv = xv.decompose();    yv=yv.decompose()
        fft_brute = np.zeros(xv.shape, dtype=complex).flatten()
        high_order = np.zeros(xv.shape, dtype=complex).flatten()
        norm = len(xv.flatten())    ##not sure if this is the proper normalization..
        R = distance
        ###This can be done faster using threads...
        for i,x,y in zip(range(len(xv.flatten())), xv.flatten(), yv.flatten()):
            exp = k*(x*uv+y*vv)
            vanilla = np.exp(1j*exp.to_value(apu.rad))*F
            fft_brute[i] = np.sum(vanilla)/norm
            high = -vanilla*1j*k*(
                uv.to_value(apu.rad)*x*(x**2+y**2)/(2*R**2)+
                vv.to_value(apu.rad)*y*(x**2+y**2)/(2*R**2)-
                (uv.to_value(apu.rad))**2*x**2/(2*R)-
                (vv.to_value(apu.rad))**2*y**2/(2*R)-
                uv.to_value(apu.rad)*vv.to_value(apu.rad)*x*y/R
                )
            high_order[i] = np.sum(high.decompose())/norm
        fft_brute = fft_brute.reshape(xv.shape)
        high_order = high_order.reshape(xv.shape)

    E_shift = fft_brute+high_order
    x0 = int((phase_zero[0]/(x_shift[1]-x_shift[0])).decompose())
    y0 = int((phase_zero[1]/(y_shift[1]-y_shift[0])).decompose())
    k = np.exp(-1j*np.angle(E_shift[len(x_shift)//2+x0,
                                    len(y_shift)//2+y0]))
    E_shift = E_shift*k
    E_shift = E_shift/np.max(np.abs(E_shift))
    #E_shift = E_shift[::-1,:]   ##there is a weird rotation, maybe in the fftshifts...for FFT2
    E_shift = E_shift[:,::-1]   ##when doing IFFT!!
    fft_brute =  fft_brute[:,::-1]
    high_order = high_order[:,::-1]
    return E_shift, x_shift.to(apu.m), y_shift.to(apu.m), fft_brute, high_order


def high_term_worker(xv, yv,indices, F,
                     uv,vv, k, ds, out_vanilla,out_high,
                     shape,
                     R=1835*apu.m):
    """
    xv,yv: meshgrids where you want to compute the aperture. These are in mts
    uv,vv: meshgrids of the measurement beam map. These are in angular units
    """
    out_vanilla_local = np.frombuffer(out_vanilla, dtype=np.complex128).reshape(shape)
    out_high_local = np.frombuffer(out_high, dtype=np.complex128).reshape(shape)
    for i in indices:
        x = xv[i]; y=yv[i]
        exp = k*(x*uv+y*vv)
        val = np.exp(1j*exp.to_value(apu.rad))*F
        out_vanilla_local[i] = np.sum(val)*ds
        high = -val*1j*k*(
                uv.to_value(apu.rad)*(x*(x**2+y**2))/(2*R**2)+
                vv.to_value(apu.rad)*(y*(x**2+y**2))/(2*R**2)-
                uv.to_value(apu.rad)**2*x**2/(2*R)-
                vv.to_value(apu.rad)**2*y**2/(2*R)-
                uv.to_value(apu.rad)*vv.to_value(apu.rad)*x*y/R)
        out_high_local[i] = np.sum(high)*ds


def fft_high_term_batch(uv,vv, F, wavel, R=1835*apu.m, 
                        max_threads=6, batch_size=64):
    k = 2*np.pi/wavel
    shape = F.shape[0]*F.shape[1]
    vanilla = multiprocessing.Array('d', 2*shape, lock=False)
    high = multiprocessing.Array('d', 2*shape, lock=False)
    ##we compute the x,y griding as the FFT
    u = uv[0,:]; v=vv[:,0]
    du = u[1]-u[0]; dv=v[1]-v[0]
    ds = du.to_value(apu.rad)*dv.to_value(apu.rad)
    x,y = np.fft.fftfreq(len(u), du), np.fft.fftfreq(len(v), dv)
    x_shift = np.fft.fftshift(x) * wavel * apu.rad
    y_shift = np.fft.fftshift(y) * wavel * apu.rad
    xv, yv = np.meshgrid(x_shift, y_shift)
    xv = xv.decompose().flatten()
    yv = yv.decompose().flatten()
    ###
    batches = xv.shape[0]//batch_size
    remains = xv.shape[0]%batch_size
    i=0
    workers = []
    while(i<batches):
        if(len(workers)< max_threads):
            indices = np.arange(batch_size)+batch_size*i
            #ipdb.set_trace()
            #test
            #high_term_worker(xv,yv,indices,F,uv,vv,k,ds,vanilla, high, shape, R)
            proc = multiprocessing.Process(target=high_term_worker,
                                           args=(xv,
                                                 yv,
                                                 indices,
                                                 F,
                                                 uv,
                                                 vv,
                                                 k,
                                                 ds,
                                                 vanilla,
                                                 high,
                                                 shape,
                                                 R
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
    print("joining last threads")
    for j in range(len(workers)):
        proc = workers.pop()
        try:
            proc.join()
        except:
            print("Error joining thread!")
    print("All thread joinsed")
    if(remains !=0):
        print("running the remained part")
        indices = np.arange(remains)+batch_size*i
        high_term_worker(xv,
                         yv,
                         indices,
                         F,
                         uv,
                         vv,
                         k,
                         ds,
                         vanilla,
                         high,
                         shape,
                         R)
    high = np.frombuffer(high, dtype=np.complex128).reshape(F.shape)
    vanilla = np.frombuffer(vanilla, dtype=np.complex128).reshape(F.shape)
    xv = xv.reshape(F.shape)
    yv = yv.reshape(F.shape)
    return high, vanilla, xv, yv





def BeamPattern_to_Aperture_classical(F , u, v, wavel, phase_zero=[2.5*apu.m,2.5*apu.m]):
    """
    Plain fft to pass from beam pattern to aperture.. Note that since we made 
    nearfield holography the aperture have high order terms that need to be 
    corrected.
    F       :   measured beam pattern
    u       :   array in apu.deg
    v       :
    wavel   :   wavelength
    """
    #normalize the beam
    F = F/np.max(np.abs(F))
    F_shift = np.fft.ifftshift(F)
    E = np.fft.ifft2(F_shift)
    E_shift = np.fft.fftshift(E)    ##fix the shift of fft implementation, old..
    #E_shift = np.fliplr(np.flipud(E_shift))
    ##set the phase=0
    du = u[1]-u[0]
    dv = v[1]-v[0]
    x,y = np.fft.fftfreq(len(u), du), np.fft.fftfreq(len(v), dv)
    #workaround to get the right units 
    if type(u) == apu.quantity.Quantity:
        if astropy.__version__ < '4':
            x_shift = np.fft.fftshift(x) * x.unit * wavel * apu.rad
            y_shift = np.fft.fftshift(y) * y.unit * wavel * apu.rad
        else:
            x_shift = np.fft.fftshift(x) * wavel * apu.rad
            y_shift = np.fft.fftshift(y) * wavel * apu.rad
    else:
        x_shift = np.fft.fftshift(x) * wavel.to_value(apu.m)
        y_shift = np.fft.fftshift(y) * wavel.to_value(apu.m)

    ##set the zero phase in some point
    ###TODO!!! It seems to be super sensible to this!
    ##we place the zero phase at some region away from the center
    x0 = int((phase_zero[0]/(x_shift[1]-x_shift[0])).decompose())
    y0 = int((phase_zero[1]/(y_shift[1]-y_shift[0])).decompose())
    k = np.exp(-1j*np.angle(E_shift[len(x_shift)//2+x0,
                                    len(y_shift)//2+y0]))
    fft_data = E_shift
    E_shift = E_shift*k
    E_shift = E_shift/np.max(np.abs(E_shift))
    #E_shift = E_shift[::-1,:]   ##there is a weird rotation, maybe in the fftshifts...for FFT2 -->cornetoscopio
    E_shift = E_shift[:,::-1]   ##when doing IFFT!!
    return E_shift, x_shift.to(apu.m), y_shift.to(apu.m), fft_data, 0


def Aperture_to_BeamPattern(E,x,y,wavel):
    """
    """
    E_shift = np.fft.ifftshift(E)
    F = np.fft.ifft2(E_shift)
    F_shift = np.fft.fftshift(F)    ##fix the shift of fft implementation, old..
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    u,v = np.fft.fftfreq(len(x), dx), np.fft.fftfreq(len(y), dy)


    if type(x) == apu.quantity.Quantity:
        if astropy.__version__ < '4':
            u_shift = np.fft.fftshift(u) * u.unit * wavel * apu.rad
            v_shift = np.fft.fftshift(v) * v.unit * wavel * apu.rad
        else:
            u_shift = np.fft.fftshift(u) * wavel * apu.rad
            v_shift = np.fft.fftshift(u) * wavel * apu.rad
    else:
        u_shift = np.fft.fftshift(u) * wavel.to_value(apu.m)
        v_shift = np.fft.fftshift(v) * wavel.to_value(apu.m)
    F_shift = F_shift[:,::-1]
    return F_shift, u_shift, v_shift









def Aperture_tilt(E,xv,yv, tilt_x, tilt_y, wavel, debug=False):
    """
    tilt_x  :   tilt in angle units
    tilt_y  :   tilt in angle units
    """
    if(debug):
        print("Debug Aperture_tilt")
        step = xv[0,1]-xv[0,0]
        N = xv.shape[0]
        X = np.arange(N)
        xv, yv = step*np.meshgrid(X,X)
    ###Here now I have doubts about why there is tangent here..
    x_angle = np.tan(tilt_x.to_value(apu.rad))*2*np.pi/wavel.to_value(apu.m)
    y_angle = np.tan(tilt_y.to_value(apu.rad))*2*np.pi/wavel.to_value(apu.m)
    theta_x = xv.to_value(apu.m)*x_angle
    theta_y = yv.to_value(apu.m)*y_angle
    out = E*np.exp(-1j*(theta_x+theta_y))
    return out



def Nearfield_defocus_fitting(E, x,y, wavel, mask, distance=1835*apu.m, f_prim=4.8*apu.m, 
                parameters=[-15.208, 0,0], debug=False
                            ):
    """
    The idea is to find the defocus value via an optimization that search the defocus 
    value that genreates the lower RMS in the antenna surface. When having the 
    best value its given to the DefocusCorrection function
    E           :   Aperture 
    x,y         :   array in apu.m
    panel_mask  :   mask to not consider certain zones of the surface (like external rings, blocked areas, etc)
    distance    :   distance from the test source to the antenna in apu.m
    parameters  :   parameters to fit: [defocus (in mm), tilt_x (in deg), tilt_y (in deg)]
    """
    if(debug):
        print("debug Nerfield_defocus_fitting")
        step = x[1]-x[0]; N=len(x)
        X = np.linspace(-step*(N-1)/2,step*(N-1)/2,N)
        xv, yv = np.meshgrid(X,X)
    else:
        xv, yv = np.meshgrid(x,y)

    def fit_function(params,xv,yv, wavel, distance, f_prim):
        r2 = xv**2+yv**2
        correct = DefocusCorrection(E, params[0]*apu.mm, wavel, xv,yv, distance, f_prim)
        tilt = Aperture_tilt(correct, xv,yv,params[1]*apu.deg, params[2]*apu.deg, wavel)
        surf = Aperture_to_Surface(tilt, xv,yv, wavel, f_prim)
        rms = calculate_surface_rms(surf, mask)
        return rms
    res = minimize(fun=fit_function, 
                   x0=parameters,
                   args=(
                       xv,
                       yv,
                       wavel,
                       distance,
                       f_prim,
                       ),
                   method='Powell',
                   options={'ftol': 0.1, 'xtol': 1e-3,'disp': True, 'maxfev':300})
    #options={'ftol': 0.1, 'xtol': 1e-3,'disp': True, 'maxfev':300})
    return res



def DefocusCorrection(E, defocus, wavel, xv, yv,distance=1835*apu.m, f_prim=4.8*apu.m, debug=False):
    """
    This function corrects the defocus and the nearfield
        nearfield term: r**2/(2*dist)-(r**4)/(8*distance**3)
        defoucs term:   r**2
    Check Baars paper of nearfield holo.. these are the term outside of the awfull integral

    defocus:    defocus in distance units
    """
    if(debug):
        print('debug DefocusCorrection')
        step = xv[0,1]-xv[0,0]; N = xv.shape[0]
        X = np.linspace(-step*(N-1)/2, step*(N-1)/2,N)
        xv, yv = np.meshgrid(X,X)
    r2 = xv**2+yv**2
    df = -defocus
    k = 2*np.pi/wavel.to_value(apu.m)
    delta_p1 = r2/(2*distance)-r2**2/(8*distance**3)
    delta_p2 = (r2+((f_prim+df)-r2/(4*f_prim))**2)**0.5-((f_prim+df)+r2/(4*f_prim))
    out = E*np.exp(-1j*k*(delta_p1.to_value(apu.m)+(delta_p2.to_value(apu.m))))
    return out





