import numpy as np
import matplotlib.pyplot as plt
from holo_utils import *
from large_scale import *
from holo_fourier import Aperture_to_BeamPattern, BeamPattern_to_Aperture


filename = '/home/seba/Workspace/apex/holo_codes_back/data/20260502_20365000/output.reg'
freq = 92.4*apu.GHz
pol_degree = 2
f_prim = 4.8*apu.m

###
###
###

wavel = cte.c/freq
F,u,v = read_regrid_data(filename)
F_corrected = phase_correction(F,u,v,wavel)
F_corrected = F_corrected/np.max(np.abs(F_corrected))*np.exp(-1j*np.angle(F_corrected[128,128]))

np.savez('test.npz', E=F_corrected)



E,x,y,_,_ = BeamPattern_to_Aperture(F_corrected,u[0,:],v[:,0], wavel, phase_zero=[0*apu.m, 0*apu.m])
xv,yv = np.meshgrid(x,y)

#discount global effects.. 

#We are going to look just at the phase in the aperture, and do dirty tricks to use the functions
mask, c_mask, p_mask, r_mask = APEX_mask(xv,yv)
phase_corr, phase_model, params = polynomial_large_scale_removal(pol_degree, np.angle(E)*apu.m, mask, xv,yv)

E_corr = np.abs(E)*np.exp(1j*phase_corr.to_value(apu.m))

F_new, u_new, v_new = Aperture_to_BeamPattern(E_corr, x,y,wavel)





####Here I translate first to the surface error, fit there and then return to the aperture-> beam patter
###the results are not good!
#surf = Aperture_to_Surface(F,xv,yv,wavel, f_prim)
#pol_corr, pol_model, params = polynomial_large_scale_removal(pol_degree, surf, mask, xv, yv)
##now we need to go back
#ap_corr = Surface_to_Aperture(surf, xv**2+yv**2,wavel, f_prim)  ##this is the phase only!
#A = np.abs(E)*ap_corr
#field_cook, u_c, uv = Aperture_to_BeamPattern(A,x,y, wavel)







