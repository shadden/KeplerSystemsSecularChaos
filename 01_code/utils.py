import numpy as np

def resonance_crosses_Pratio_region(kvec,Jin,Jout):
    r"""
    Determine whether the three-body resonance defined by :math:`\mathbf{k}\cdot\mathbf{n}` intersects the region in period-ratio space defined by
    .. math::
        \begin{align}
        \frac{J_\mathrm{in} - 1}{J_\mathrm{in}} &< n_2/n_1 < \frac{J_\mathrm{in}}{J_\mathrm{in} + 1}\\
        \frac{J_\mathrm{out} - 1}{J_\mathrm{out}} &< n_3/n_2 < \frac{J_\mathrm{out}}{J_\mathrm{out} + 1}        
        \end{align}
    
    Parameters
    ----------
    kvec : ndarray
        Array  of 3BR integeger coefficeints
    Jin : int
        Defines the bounding resonances of the inner planet pair
    Jout : int
        Defines the boudning resonances of the outer planet pair

    Returns
    -------
    bool
        True if the specified 3BR crosses the period-ratio region.
    """
    xmin,xmax = (Jin-1)/Jin,Jin/(Jin+1)
    ymin,ymax = (Jout-1)/Jout,Jout/(Jout+1)
    k1,k2,k3 = kvec
    res_y2x = lambda y: k1/(k2 + k3 * y)
    xres_min,xres_max = np.sort([res_y2x(y) for y in (ymin,ymax)])
    return not (xres_max < xmin or xres_min > xmax)