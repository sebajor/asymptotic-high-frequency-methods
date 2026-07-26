import numpy as np
import matplotlib.pyplot as plt
from astropy import units as apu
from astropy import constants as cte
import astropy
import os, yaml
import ipdb
from scipy.optimize import minimize
from scipy import optimize
from math import factorial as f
from astropy.modeling import models, fitting
from holo_fourier import Aperture_to_Surface
import warnings

"""
After having the nearfield effects removed and with the aperture fields you 
have to account for the large scale errors in the plate.
Here we have two methods, the standard one consisting of fitting of the Zernike
polynomials in the aperture and a polynimial fit in the surface of the primary.
"""
###
##Zernike way, based on the pyoof implementation
###
def illum_parabolic(xv, yv, I_coeff, pr):
    [i_amp, c_dB, q, x0, y0] = I_coeff
    # workaround for units
    if type(c_dB) == apu.quantity.Quantity:
        c = 10 ** (c_dB / 20. / apu.dB)
    else:
        c = 10 ** (c_dB / 20.)
    if type(x0) != apu.quantity.Quantity:
        x0 *= apu.m
    if type(y0) != apu.quantity.Quantity:
        y0 *= apu.m

    # c_dB has to be negative, bounds given [-8, -25]
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    # Parabolic taper on a pedestal
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        Ea = np.nan_to_num(
            (i_amp * (c + (1. - c) * (1. - (r / pr) ** 2) ** q)).value
            )
        # some values of c_dB may introduce np.nan in the cross terms
    return Ea

def illum_gauss(xv, yv, I_coeff, pr):
    _amp, c_dB = I_coeff[:2]
    x0, y0 = I_coeff[-2:]

    # workaround for units
    if type(c_dB) == apu.quantity.Quantity:
        sigma = 10 ** (c_dB / 20. / apu.dB)
    else:
        sigma = 10 ** (c_dB / 20.)
    if type(x0) != apu.quantity.Quantity:
        x0 *= apu.m
    if type(y0) != apu.quantity.Quantity:
        y0 *= apu.m

    Ea = (
        i_amp * np.sqrt(2 * np.pi * sigma ** 2) *
        np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma * pr) ** 2))
        ).value
    return Ea

def illum_uniform(xv,yv, I_coeff, pr):
    _amp, c_dB = I_coeff[:2]
    out = _amp*np.ones(xv.shape)
    return out


def radial_zernike(n,m, rho):
    """
    n = radial order
    m = phase order
    rho = normaliz radial values np.sqrt(x**2+y**2)
    """
    a = (n + m) // 2
    b = (n - m) // 2

    radial_poly = sum(
        (-1) ** s * f(n - s) * rho ** (n - 2 * s) /
        (f(s) * f(a - s) * f(b - s))
        for s in range(0, b + 1)
        )

    return radial_poly

def Zernike_circle_generator(n,l, rho, theta):
    """
    """
    m = abs(l)
    radial = radial_zernike(n,m,rho)
    if(l<0):
        zernike_circle_poly = radial*np.sin(m*theta)
    else:
        zernike_circle_poly = radial*np.cos(m*theta)
    return zernike_circle_poly


def Zernike_aperture(K_coeff, I_coeff, wavel, illum_func, xv, yv, pr):
    """
    K_coeffs:   
    illum_func: 
    xv, yv:     
    pr: primary radius
    """
    r = np.sqrt(xv**2+yv**2)
    rho = (r/pr).decompose()
    theta = np.arctan2(yv,xv)
    # Total number of Zernike circle polynomials
    n = int((np.sqrt(1 + 8 * len(K_coeff)) - 3) / 2)
    # list of tuples with (n, l) allowed values
    nl = [(i, j) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]
    zernike_aberration = sum(
        K_coeff[i] * Zernike_circle_generator(*nl[i], rho, theta)
        for i in range(len(K_coeff))
        ).value
    zernike_aberration[(r**2)>(pr**2)] = 0
    phi = zernike_aberration*2*np.pi/wavel.to_value(apu.m)
    #ipdb.set_trace()
    #illumination
    Ea = illum_func(xv=xv, yv=yv, I_coeff=I_coeff, pr=pr)
    out = Ea*np.exp(phi*1j)
    return out

def Zernike_fit_aux(params, data, mask, wavel, illum_func, xv,yv,pr, illum_cte=None):
    I_coeffs, K_coeffs = params[:4], params[4:]
    K_coeffs = K_coeffs*wavel.to_value(apu.m)
    if(illum_cte is not None):
        I_coeffs = illum_cte
    model = Zernike_aperture(K_coeffs, I_coeffs, wavel, illum_func, xv, yv, pr)
    error = np.abs(model[mask]-data[mask])
    return error



def Zernike_fit(n, data, mask, wavel, illum_func, xv,yv,pr, illum_cte=None):
    """
    """
    I_coeffs = np.ones(4).tolist() #np.random.normal(0,0.3, 4).tolist()
    N_K_coeff = (n + 1) * (n + 2) // 2
    K_coeff = np.random.normal(0., .08, N_K_coeff).tolist()
    params0 = I_coeffs+K_coeff
    res_lsq = optimize.least_squares(
            fun=Zernike_fit_aux,
            x0=params0,
            args=(
                data, 
                mask, 
                wavel,
                illum_func,
                xv,
                yv,
                pr,
                illum_cte
                )
            )
    return res_lsq



def Zernike_large_scale_removal(n, data, mask, wavel, 
                                illum_func, xv, yv,pr,
                                f_prim, illum_cte=None
                                ):
    """
    I am not sure what is best Aperture-fit or doing it in the surface...
    data:   Aperture data
    """
    r2 = xv**2+yv**2
    surf = Aperture_to_Surface(data, xv,yv, wavel, f_prim)
    params = Zernike_fit(n, data, mask, wavel, illum_func, xv,yv,pr, illum_cte=illum_cte)
    I_coeff = params.x[:4]
    K_coeff = params.x[4:]*wavel.to_value(apu.m)
    fit = Zernike_aperture(K_coeff, I_coeff, wavel, illum_func, xv,yv,pr)
    surf_fit =  Aperture_to_Surface(fit, xv, yv, wavel, f_prim)
    out = surf-surf_fit
    return out, surf_fit, I_coeff, K_coeff





###
### polynomial fit 
###

def polynomial_large_scale_removal(n, data,mask,xv,yv):
    """
    data: surface error data
    """
    xv = xv.to_value(apu.m)
    yv = yv.to_value(apu.m)
    data = data.to_value(apu.m)
    p_init = models.Polynomial2D(n)
    fit_p = fitting.LevMarLSQFitter()
    with warnings.catch_warnings(): # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, xv[mask], yv[mask], data[mask])
    surf_fit = p(xv, yv)*apu.m
    out = data*apu.m-surf_fit
    params = p.parameters   ##this parameters are enumerated in the worst possible option, you could check the namings in p.param_names
    return out, surf_fit, params








