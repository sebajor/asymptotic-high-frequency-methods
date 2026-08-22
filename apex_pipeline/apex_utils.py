import numpy as np
import matplotlib.pyplot as plt

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



    
