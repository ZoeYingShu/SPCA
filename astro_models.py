import numpy as np
import batman


def transit_model(time, t0, per, rp, a, inc, ecc, w, u1, u2):
    params = batman.TransitParams()                      #object to store transit parameters
    params.t0 = t0                                       #time of inferior conjunction
    params.per = per                                     #orbital period
    params.rp = rp                                       #planet radius (in units of stellar radii)
    params.a = a                                         #semi-major axis (in units of stellar radii)
    params.inc = inc                                     #orbital inclination (in degrees)
    params.ecc = ecc                                     #eccentricity
    params.w = w                                         #longitude of periastron (in degrees)
    params.limb_dark = "quadratic"                       #limb darkening model
    params.u = [u1, u2]                                  #limb darkening coefficients

    m = batman.TransitModel(params, time)                #initializes model
    flux = m.light_curve(params)
    t_secondary = m.get_t_secondary(params)
    anom       = m.get_true_anomaly()
    return flux, t_secondary, anom

def eclipse(time, t0, per, rp, a, inc, ecc, w, u1, u2, fp, t_sec):
    params = batman.TransitParams()                      #object to store transit parameters
    params.t0 = t0                                       #time of inferior conjunction
    params.per = per                                     #orbital period
    params.rp = rp                                       #planet radius (in units of stellar radii)
    params.a = a                                         #semi-major axis (in units of stellar radii)
    params.inc = inc                                     #orbital inclination (in degrees)
    params.ecc = ecc                                     #eccentricity
    params.w = w                                         #longitude of periastron (in degrees)
    params.limb_dark = "quadratic"                       #limb darkening model
    params.u = [u1, u2]                                  #limb darkening coefficients
    params.fp = fp                                       #planet/star brightnes
    params.t_secondary = t_sec
    
    m = batman.TransitModel(params, time, transittype="secondary")  #initializes model
    flux = m.light_curve(params)
    return flux

def area(time, t_sec, per, rp, inc_raw, p0_phase, mode):
    #FOR NOW I AM ASSUMING ONLY 1st ORDER VARIATIONS PERMITTED WITH AREA VARIATIONS!!!
    r2 = p0_phase[2]
    
    t = time - t_sec
    w = 2*np.pi/per
    phi = (w*t-np.pi)%(2*np.pi)
    inc = inc_raw*np.pi/180
    #R = np.array([[np.sin(inc)*np.cos(phi),   np.sin(phi),  np.cos(inc)*np.cos(phi)],
    #              [-np.sin(inc)*np.sin(phi),  np.cos(phi),  -np.cos(inc)*np.sin(phi)],
    #              [-np.cos(inc),              0,            np.sin(inc)]])
    R = np.zeros((len(phi),3,3))
    R[:,0,0] = np.sin(inc)*np.cos(phi)
    R[:,0,1] = np.sin(phi)
    R[:,0,2] = np.cos(inc)*np.cos(phi)
    R[:,1,0] = -np.sin(inc)*np.sin(phi)
    R[:,1,1] = np.cos(phi)
    R[:,1,2] = -np.cos(inc)*np.sin(phi)
    R[:,2,0] = -np.cos(inc)
    R[:,2,2] = np.sin(inc)
    a_mat = np.array([[1/r2**2,  0,        0],
                      [0,        1/rp**2,  0],
                      [0,        0,        1/rp**2]])[np.newaxis,:,:]
    #[[a, d, f],
    # [_, b, e],
    # [_, _, c]] 
    arr = np.matmul(R.transpose(0,2,1), np.matmul(a_mat,R))
    a = arr[:,0,0]
    b = arr[:,1,1]
    c = arr[:,2,2]
    d = arr[:,0,1]
    e = arr[:,1,2]
    f = arr[:,0,2]
    return np.pi/np.sqrt(3*b*f**2/a + 3*c*d**2/a + -6*d*e*f/a + b*c - e**2)/(np.pi*rp**2)

def phase_variation(time, t_sec, per, anom, w, p0_phase, mode):
    if 'eccent' in mode:
        phi = anom + w + np.pi/2
    else:
        t = time - t_sec
        w = 2*np.pi/per
        phi = (w*t)
    if 'v2' in mode:
        A, B, C, D = p0_phase[:4]
        phase = 1 + (A*(np.cos(phi)-1) + (B*np.sin(phi))) + C*(np.cos(2*phi)-1) + (D*np.sin(2*phi))
    else:
        A, B = p0_phase[:2]
        phase = 1 + (A*(np.cos(phi)-1) + (B*np.sin(phi)))
    return phase

def fplanet_model(time, anom, t0, per, rp, a, inc, ecc, w, u1, u2, fp, t_sec, p0_phase, mode):
    phase = phase_variation(time, t_sec, per, anom, w, p0_phase, mode)
    eclip = eclipse(time, t0, per, rp, a, inc, ecc, w, u1, u2, fp, t_sec)
    if 'ellipsoid' in mode:
        sArea = area(time, t_sec, per, rp, inc, p0_phase, mode)
    else:
        sArea = 1
    return sArea*phase*(eclip - 1)

def ideal_lightcurve(time, p0, per, mode):
    t0, rp, a, inc, ecosw, esinw, q1, q2, fp = p0[:9]
    
    ecc = np.sqrt(ecosw**2 + esinw**2)
    w   = np.arctan2(esinw, ecosw)
    u1  = 2*np.sqrt(q1)*q2
    u2  = np.sqrt(q1)*(1-2*q2)
    # create transit first and use orbital paramater to get time of superior conjunction
    transit, t_sec, anom = transit_model(time, t0, per, rp, a, inc, ecc, w, u1, u2)
    
    #ugly way of doing this as might pick up detector parameters, but thats alright - faster this way and still safe
    p0_phase = p0[9:13]
    fplanet = fplanet_model(time, anom, t0, per, rp, a, inc, ecc, w, u1, u2, fp, t_sec, p0_phase, mode)
    
    # add both light curves
    f_total = transit + fplanet
    return f_total

def check_phase(time, p0, per, mode):
    t0, rp, a, inc, ecosw, esinw, q1, q2, fp = p0[:9]
    #ugly way of doing this as might pick up detector parameters, but thats alright - faster this way and still safe
    p0_phase = p0[9:13]
    
    ecc = np.sqrt(ecosw**2 + esinw**2)
    w   = np.arctan2(esinw, ecosw)
    
    params = batman.TransitParams()                      #object to store transit parameters
    params.t0 = t0                                       #time of inferior conjunction
    params.per = per                                     #orbital period
    params.rp = rp                                       #planet radius (in units of stellar radii)
    params.a = a                                         #semi-major axis (in units of stellar radii)
    params.inc = inc                                     #orbital inclination (in degrees)
    params.ecc = ecc                                     #eccentricity
    params.w = w                                         #longitude of periastron (in degrees)
    params.limb_dark = "uniform"                         #limb darkening model
    params.u = []                                        #limb darkening coefficients
    params.fp = fp                                       #planet/star brightnes
    
    m = batman.TransitModel(params, time)                #initializes model
    t_sec = m.get_t_secondary(params)
    params.t_secondary = t_sec
    m = batman.TransitModel(params, time, transittype="secondary")  #initializes model
    anom = m.get_true_anomaly()
    flux = m.light_curve(params)
    
    phase = phase_variation(time, t_sec, per, anom, w, p0_phase, mode)
    if 'ellipsoid' in mode:
        sArea = area(time, t_sec, per, rp, inc, p0_phase, mode)
    else:
        sArea = 1
    return np.any((sArea*phase*(flux-1)) < 0)