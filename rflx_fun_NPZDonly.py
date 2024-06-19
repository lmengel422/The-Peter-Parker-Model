"""
Functions for the rflx code.
"""

import numpy as np

# function to create Sin and Sout
def get_Sio_chatwin(Socn, ds, nx, L=50e3): #50e3
    a = Socn/(L**1.5)
    alpha = ds/L
    x = np.linspace((alpha/(2*a))**2,L,nx)
    Sin = a*x**1.5 + alpha*x/2
    Sout = a*x**1.5 - alpha*x/2
    return Sin, Sout, x, L

def get_Sio_fjord(Socn, ds, nx, L=100e3):
    x = np.linspace(0,L,nx)
    Sin = np.linspace(Socn - ds, Socn, nx)
    Sout = np.linspace(Socn - 2*ds, Socn - ds, nx)
    return Sin, Sout, x, L

def a_calc(Sin, Sout):
    Sin1 = Sin[1:] #Everthing except first value (index 0)
    Sin0 = Sin[:-1] #Everthing except last value
    Sout1 = Sout[1:]
    Sout0 = Sout[:-1]
    # Calculate Efflux-Reflux fractions (a1 and a0), at xm
    # (notation different from Cokelet and Stewart 1985)
    a0 = (Sout0/Sin0)*(Sin1-Sin0)/(Sin1-Sout0) # reflux - down
    a1 = (Sin1/Sout1)*(Sout1-Sout0)/(Sin1-Sout0) # efflux - up
    return a0, a1
    
def get_time_step(dx, Qout, B, hs, ndays):
    """
    Calculate the time step dynamically using some factor (0.9) times
    the shortest time for water to traverse a grid box in the
    along_channel direction (experiments showed that it was unstable
    for cfl_factor > 1.1).
    """
    cfl_factor = 0.9
    dt = cfl_factor * np.min(dx/(Qout[1:]/(B*hs))) # time step (s)
    NT = int(ndays*86400/dt) # total number of time steps
    NS = 100 # number of saves 10
    return dt, NT, NS

def c_calc(csp, cdp, info_tup, riv=0, ocn=0, Ts=np.inf, Td=np.inf, Cs=1, Cd=0):
    """
    This is the main computational engine for the time dependent solution.
    
    csp, cdp = vectors [xm] of initial tracer concentrations in the two layers
    csa, cda = arrays [time, xm] of the surface and deep concentrations
        over the course of the simulation
    
    info_tup = tuple of properties defining the grid, timestep and circulation
    
    riv = value of tracer coming in from river
    ocn = value of tracer coming in from ocean
    Ts = relaxation timescale [days] for surface layer
    Td = relaxation timescale [days] for deep layer
    Cs = value of tracer to relax to in surface layer
    Cd = value of tracer to relax to in deep layer
    """
    # unpack some parameters
    NS, NX, NT, dt, dvs, dvd, Qout, Qin, a0, a1, qr, way = info_tup
    # Get Q's at box edges used in the box model.
    # NOTE: all Q's are positive for the box model, whereas
    # Qout is negative in my usual TEF sign convention.
    Qout0 = Qout[:-1]
    Qin0 = Qin[:-1]
    Qout1 = Qout[1:]
    Qin1 = Qin[1:]
    NTs = int(NT/NS) # save interval, in timesteps
    # initialize arrays to save in
    csa = np.nan * np.ones((NS,NX-1))
    cda = np.nan * np.ones((NS,NX-1))
    times = np.nan * np.ones((NS))
    tta = 0 # index for periodic saves
    for tt in range(NT):
        # boundary conditions 
        Cout0 = np.concatenate(([riv], csp[:-1])) # river 
        Cin1 = np.concatenate((cdp[1:], [ocn])) # ocean 
        # update fields
        cs = csp + (dt/dvs)*(Qout0*(1-a0)*Cout0 + Qin1*a1*Cin1 - Qout1*csp + qr*riv) + (Cs-csp)*(dt/86400)/Ts 
        cd = cdp + (dt/dvd)*( Qin1*(1-a1)*Cin1 + Qout0*a0*Cout0 - Qin0*cdp) + (Cd-cdp)*(dt/86400)/Td
        cs[cs<0] = 0
        cd[cd<0] = 0
        if way == 'old':#old way
            cd[0] = cd[1] # this helps when using riv = ocn = const.
        else: #new way
            cd[0] = np.nan # first bottom cell not active, so mask
        csp = cs.copy()
        cdp = cd.copy()
        if (np.mod(tt, NTs) == 0) and tta < NS:
            # periodic save
            csa[tta,:] = cs
            cda[tta,:] = cd
            times[tta] = tt
            tta += 1
    return csa, cda, times

def nut_sal(Socn): #Table 2 Davis 2014
    S = Socn
    if S < 31.9:
        N=0
    elif S < 33:
        N=20.15*S-642.8
    elif S < 33.82:
        N=9.59*S-294.3
    elif S < 34.25:
        N=34.83*S-1148
    elif S < 34.3:
        N=45
    else:
        N=-37.3*S+1324
    Nocn = N
    return Nocn

def npzd_calc(nsp, ndp, psp, pdp, zsp, zdp, dsp, ddp, hs, hd, info_tup, riv=0, ocn=0, Nriv=0, Nocn=0, Priv=0, Pocn=0, Zriv=0, Zocn=0, Driv = 0, Docn = 0, Ts=np.inf, Td=np.inf, Ns=0, Nd=0, Ps=0, Pd=0, Zs=0, Zd=0, Ds=0, Dd=0):
    """
    This is the main computational engine for the time dependent solution for fully growing NPZD in shallow and deep.
    
    nsp, ndp = vectors [xm] of initial nutrient concentrations in the two layers
    psp, pdp = vectors [xm] of initial phytoplankton concentrations in the two layers
    zsp, zdp = vectors [xm] of initial zooplankton concentrations in the two layers
    dsp, ddp = vectors [xm] of initial detritus concentrations in the two layers
    nsa, nda = arrays [time, xm] of the surface and deep concentrations Nutrient
    psa, pda = arrays [time, xm] of the surface and deep concentrations Phytoplankton
    zsa, zda = arrays [time, xm] of the surface and deep concentrations Zooplankton
    dsa, dda = arrays [time, xm] of the surface and deep concentrations Detritus
        over the course of the simulation
    
    hs = thickness shallow layer
    hd = thickness deep layer
    info_tup = tuple of properties defining the grid, timestep and circulation
    
    riv = value of salinity coming in from river
    ocn = value of salinity coming in from ocean
    Nriv = value of nutrients coming in from river
    Driv = value of detritus coming in from river
    Nocn = value of nutrients coming in from ocean
    Pocn = value of phytoplankton coming in from ocean
    Zocn = value of zooplankton coming in from ocean
    Docn = value of detritus coming in from ocean
    Ts = relaxation timescale [days] for surface layer
    Td = relaxation timescale [days] for deep layer
    Ns = value of N to relax to in surface layer
    Nd = value of N to relax to in deep layer
    Ps = value of P to relax to in surface layer
    Pd = value of P to relax to in deep layer
    Zs = value of Z to relax to in surface layer
    Zd = value of Z to relax to in deep layer
    Ds = value of D to relax to in surface layer
    Dd = value of D to relax to in deep layer
    """
    
     # objects needed
    class sterms:
        def __init__(self,psterms):
            self.advright = psterms[:,:,0]
            self.reflux = psterms[:,:,1]
            self.efflux = psterms[:,:,2]
            self.outflow = psterms[:,:,3]
            self.growth = psterms[:,:,4]
            self.grazing = psterms[:,:,5]
            self.death = psterms[:,:,6]
            self.phys = self.advright + self.reflux + self.efflux + self.outflow
            self.bio = self.growth + self.grazing + self.death
    
    class dterms:
        def __init__(self,pdterms):
            self.advleft = pdterms[:,:,0]
            self.efflux = pdterms[:,:,1]
            self.reflux = pdterms[:,:,2]
            self.inflow = pdterms[:,:,3]
            self.growth = pdterms[:,:,4]
            self.grazing = pdterms[:,:,5]
            self.death = pdterms[:,:,6]
            self.phys = self.advleft + self.efflux + self.reflux + self.inflow
            self.bio = self.growth + self.grazing + self.death
    
    class detritus_terms:
        def __init__(self,dterms):
            self.shallow = dterms[:,:,0]
            self.deep = dterms[:,:,1]

    # functions needed
    def mu_i(E, N):
        # instantaneous phytoplankton growth rate
        mu0 = 2.2/86400 # maximum instantaneous growth rate: s-1
        ks = 4.6 # half-satuaration for nitrate uptake: uM N
        alpha = 0.07/86400 # initial slope of growth-ligh curve: (W m-2)-1 s-1    
        mu_i = mu0 * (N/(ks + N)) * (alpha * E)/np.sqrt(mu0**2 + alpha**2*E**2)
        return mu_i

    def I(P):
        # zooplankton ingestion
        I0 = 4.8/86400 # maximum ingestion rate: s-1
        Ks = 3 # half-satuaration for ingestion: uM N
        I = I0 * P**2 / (Ks**2 + P**2)
        return I
    
    def Ef(t): #t in days
        E0 = 200 # max light: W m-2
        E = (E0/2) * (1 + np.cos(2*np.pi*t))
        return E
    # unpack some parameters
    NS, NX, NT, dt, dvs, dvd, Qout, Qin, a0, a1, qr, way = info_tup
    # Get Q's at box edges used in the box model.
    # NOTE: all Q's are positive for the box model, whereas
    # Qout is negative in my usual TEF sign convention.
    Qout0 = Qout[:-1]
    Qin0 = Qin[:-1]
    Qout1 = Qout[1:]
    Qin1 = Qin[1:]
    '''
    Qout0 = np.zeros(NX-1)
    Qin0 = np.zeros(NX-1)
    Qout1 = np.zeros(NX-1)
    Qin1 = np.zeros(NX-1)
    '''
    NTs = int(NT/NS) # save interval, in timesteps
    # initialize arrays to save in
    nsa = np.nan * np.ones((NS,NX-1))
    nda = np.nan * np.ones((NS,NX-1))
    psa = np.nan * np.ones((NS,NX-1))
    pda = np.nan * np.ones((NS,NX-1))
    zsa = np.nan * np.ones((NS,NX-1))
    zda = np.nan * np.ones((NS,NX-1))
    dsa = np.nan * np.ones((NS,NX-1))
    dda = np.nan * np.ones((NS,NX-1))
    mu_is = np.nan * np.ones((NS,NX-1))
    mu_id = np.nan * np.ones((NS,NX-1))
    I_s = np.nan * np.ones((NS,NX-1))
    I_d = np.nan * np.ones((NS,NX-1))
    tta = 0 # index for periodic saves
    times = np.nan * np.ones((NS))
    t = np.nan * np.ones((NT))
    psterms = np.nan * np.ones((NS,NX-1,7))
    pdterms = np.nan * np.ones((NS,NX-1,7))
    detterms = np.nan * np.ones((NS,NX-1,2))
    t[0]=0
    
    #NPZD Model Biological parameters
    min_N=0.000001 #[uM N]
    min_P=0.000001 #[uM N]
    min_Z=0.000001 #[uM N]
    min_D=0.000001 #[uM N]
    
    m = 0.1/86400 # nongrazing mortality: s-1
    eps = 0.3 # growth efficiency
    xi = 2.0/86400 # mortality: s-1 (uM N)-1
    f_e = 0.5 # fraction of losses egested
    r = 0.1/86400 # remineralization rate s-1
    attsw = 0.13 #m-1 light attenuation by seawater
    attp = 0.018 #m-1 (uM N)-1 self shading
    ws = 8/86400 #sinking rate m s-1 8
    seq = 0 #fraction of sinking deep detritus sequestered
    mu0 = 2.2/86400 # maximum instantaneous growth rate: s-1
    I0 = 4.8/86400 # maximum ingestion rate: s-1

    #make ws a whole array
    ws_array = ws*np.ones(len(dsp))

    if way == 'new':
        # force sink = 0 in the first box because the bottom cell there is not active
        ws_array[0] = 0 
    
    for tt in range(NT):
        #'''
        # boundary conditions Dirichlet
        Nout0 = np.concatenate(([Nriv], nsp[:-1])) # river
        Nin1 = np.concatenate((ndp[1:], [Nocn])) # ocean
        Pout0 = np.concatenate(([Priv], psp[:-1])) # river - no phytoplankton
        Pin1 = np.concatenate((pdp[1:], [Pocn])) # ocean - match concentration
        Zout0 = np.concatenate(([Zriv], zsp[:-1])) # river - no zooplankton
        Zin1 = np.concatenate((zdp[1:], [Zocn])) # ocean - match concentration
        Dout0 = np.concatenate(([Driv], dsp[:-1])) # river - detritus
        Din1 = np.concatenate((ddp[1:], [Docn])) # ocean - match concentration
        '''
        # boundary conditions Neumann
        Nout0 = np.concatenate(([Nriv], nsp[:-1])) # river
        Nin1 = np.concatenate((ndp[1:], [Nocn])) # ocean
        Pout0 = np.concatenate(([psp[1]], psp[:-1])) # river - no phytoplankton
        Pin1 = np.concatenate((pdp[1:], [pdp[-2]])) # ocean - match concentration
        Zout0 = np.concatenate(([zsp[1]], zsp[:-1])) # river - no zooplankton
        Zin1 = np.concatenate((zdp[1:], [zdp[-2]])) # ocean - match concentration
        Dout0 = np.concatenate(([dsp[1]], dsp[:-1])) # river - detritus
        Din1 = np.concatenate((ddp[1:], [ddp[-2]])) # ocean - match concentration
        '''
        # update fields
        if tt>0:
            t[tt] = t[tt-1]+dt
        
        #Update light
        Es=Ef(t[tt]/86400) #Shallow E. Function expects it in days. 
        Ed=Es*np.exp(-attsw*hs-attp*psp*hs) #Deep E. Depends depth and shading shallow plankton population.
        
        #Nutrient
        ns = (nsp + (dt/dvs)*(Qout0*(1-a0)*Nout0 + Qin1*a1*Nin1 - Qout1*nsp + qr*Nriv)
        + dt*( -mu_i(Es, nsp)*psp + (1-eps)*(1-f_e)*I(psp)*zsp + r*dsp ) + (Ns-nsp)*(dt/86400)/Ts) 
        nd = (ndp + (dt/dvd)*(Qin1*(1-a1)*Nin1 + Qout0*a0*Nout0 - Qin0*ndp ) #+ seq*ws_array*(dvd/hd)*ddp 
        + dt*( -mu_i(Ed, ndp)*pdp + (1-eps)*(1-f_e)*I(pdp)*zdp + r*ddp ) + (Nd-ndp)*(dt/86400)/Td)
        ns[ns<0] = min_N
        nd[nd<0] = min_N
        if way == 'old':#old way
            nd[0] = nd[1] # this helps when using riv = ocn = const.
        else: #new way
            nd[0] = np.nan # first bottom cell not active, so mask
        
        #Phytoplankton
        ps = (psp + (dt/dvs)*(Qout0*(1-a0)*Pout0 + Qin1*a1*Pin1 - Qout1*psp + qr*Priv) 
        + dt*( mu_i(Es, nsp)*psp - I(psp)*zsp - m*psp ) 
        + (Ps-psp)*(dt/86400)/Ts) 
        pd = (pdp + (dt/dvd)*(Qin1*(1-a1)*Pin1 + Qout0*a0*Pout0 - Qin0*pdp) 
        + dt*( mu_i(Ed, ndp)*pdp - I(pdp)*zdp - m*pdp ) 
        + (Pd-pdp)*(dt/86400)/Td)
        ps[ps<0] = min_P
        pd[pd<0] = min_P
        if way == 'old':#old way
            pd[0] = pd[1] # this helps when using riv = ocn = const.
        else: #new way
            pd[0] = np.nan # first bottom cell not active, so mask
        
        #Zooplankton
        zs = (zsp + (dt/dvs)*(Qout0*(1-a0)*Zout0 + Qin1*a1*Zin1 - Qout1*zsp + qr*Zriv) 
        + dt*( eps*I(psp)*zsp - xi*zsp**2 ) 
        + (Zs-zsp)*(dt/86400)/Ts) 
        zd = (zdp + (dt/dvd)*(Qin1*(1-a1)*Zin1 + Qout0*a0*Zout0 - Qin0*zdp) 
        + dt*( eps*I(pdp)*zdp - xi*zdp**2 ) 
        + (Zd-zdp)*(dt/86400)/Td)
        zs[zs<0] = min_Z
        zd[zd<0] = min_Z
        if way == 'old':#old way
            zd[0] = zd[1] # this helps when using riv = ocn = const.
        else: #new way
            zd[0] = np.nan # first bottom cell not active, so mask
        
        #Detritus
        ds = (dsp + (dt/dvs)*(Qout0*(1-a0)*Dout0 + Qin1*a1*Din1 - Qout1*dsp + qr*Driv -ws_array*(dvs/hs)*dsp) 
        + dt*( (1-eps)*f_e*I(psp)*zsp + m*psp + xi*zsp**2 - r*dsp )   
        + (Ds-dsp)*(dt/86400)/Ts) 
        dd = (ddp + (dt/dvd)*(Qin1*(1-a1)*Din1 + Qout0*a0*Dout0 - Qin0*ddp +ws_array*(dvs/hs)*dsp - seq*ws_array*(dvd/hd)*ddp) 
        + dt*( (1-eps)*f_e*I(pdp)*zdp + m*pdp + xi*zdp**2 - r*ddp ) 
        + (Dd-ddp)*(dt/86400)/Td)        
        ds[ds<0] = min_D
        dd[dd<0] = min_D
        if way == 'old':#old way
            dd[0] = dd[1] # this helps when using riv = ocn = const.
        else: #new way
            dd[0] = np.nan # first bottom cell not active, so mask
              
        #For next run
        nsp = ns.copy()
        ndp = nd.copy()
        psp = ps.copy()
        pdp = pd.copy()
        zsp = zs.copy()
        zdp = zd.copy()
        dsp = ds.copy()
        ddp = dd.copy()
        if (np.mod(tt, NTs) == 0) and tta < NS:
            # periodic save
            nsa[tta,:] = ns
            nda[tta,:] = nd
            psa[tta,:] = ps
            pda[tta,:] = pd
            zsa[tta,:] = zs
            zda[tta,:] = zd
            dsa[tta,:] = ds
            dda[tta,:] = dd
            mu_is[tta,:] = mu_i(Es, nsp)
            mu_id[tta,:] = mu_i(Ed, ndp)
            I_s[tta,:] = I(psp)
            I_d[tta,:] = I(pdp)
            times[tta] = tt
            psterms[tta,:,0] = (1/dvs)*(Qout0*(1)*Pout0) #Advection right
            psterms[tta,:,1] = (1/dvs)*(Qout0*(-a0)*Pout0) #Reflux (down)
            psterms[tta,:,2] = (1/dvs)*(Qin1*a1*Pin1) #Efflux (up)
            psterms[tta,:,3] = (1/dvs)*(- Qout1*psp) #Outflow
            psterms[tta,:,4] = 1*(mu_i(Es, nsp)*psp) #Growth
            psterms[tta,:,5] =1*(- I(psp)*zsp) #Grazing
            psterms[tta,:,6] = 1*(- m*psp) #Death
            pdterms[tta,:,0] = (1/dvd)*(Qin1*(1)*Pin1) #Advection left
            pdterms[tta,:,1] = (1/dvd)*(Qin1*(-a1)*Pin1) #Efflux (up) 
            pdterms[tta,:,2] = (1/dvd)*(Qout0*a0*Pout0) #Reflux (down)
            pdterms[tta,:,3] = (1/dvd)*(- Qin0*pdp) #Inflow
            pdterms[tta,:,4] = 1*(mu_i(Ed, ndp)*pdp) #Growth 
            pdterms[tta,:,5] = 1*(- I(pdp)*zdp) #Grazing
            pdterms[tta,:,6] = 1*(- m*pdp) #Death
            detterms[tta,:,0] = (1-eps)*f_e*I(psp)*zsp + m*psp + xi*zsp**2 - r*dsp #Shallow D bio terms
            detterms[tta,:,1] = (1-eps)*f_e*I(pdp)*zdp + m*pdp + xi*zdp**2 - r*ddp #Deep D bio terms
            tta += 1
    
    
    psterms_obj = sterms(psterms)
    pdterms_obj = dterms(pdterms)
    dterms_obj = detritus_terms(detterms)

    return nsa, nda, psa, pda, zsa, zda, dsa, dda, times, psterms_obj, pdterms_obj, dterms_obj, mu_is, mu_id, I_s, I_d

def npzd_calc_change_ecol(nsp, ndp, psp, pdp, zsp, zdp, dsp, ddp, hs, hd, info_tup, riv=0, ocn=0, Nocn=0, Priv=0, Pocn=0, Zriv=0, Zocn=0, Driv = 0, Docn = 0, Ts=np.inf, Td=np.inf, Ns=0, Nd=0, Ps=0, Pd=0, Zs=0, Zd=0, Ds=0, Dd=0, ws=8/86400, Nriv=5, mu0=2.2, I0 = 4.8, m=0.1, r=0.1):
    """
    This is the main computational engine for the time dependent solution for fully growing NPZD in shallow and deep.
    
    nsp, ndp = vectors [xm] of initial nutrient concentrations in the two layers
    psp, pdp = vectors [xm] of initial phytoplankton concentrations in the two layers
    zsp, zdp = vectors [xm] of initial zooplankton concentrations in the two layers
    dsp, ddp = vectors [xm] of initial detritus concentrations in the two layers
    nsa, nda = arrays [time, xm] of the surface and deep concentrations Nutrient
    psa, pda = arrays [time, xm] of the surface and deep concentrations Phytoplankton
    zsa, zda = arrays [time, xm] of the surface and deep concentrations Zooplankton
    dsa, dda = arrays [time, xm] of the surface and deep concentrations Detritus
        over the course of the simulation
    
    hs = thickness shallow layer
    hd = thickness deep layer
    info_tup = tuple of properties defining the grid, timestep and circulation
    
    riv = value of salinity coming in from river
    ocn = value of salinity coming in from ocean
    Nriv = value of nutrients coming in from river
    Driv = value of detritus coming in from river
    Nocn = value of nutrients coming in from ocean
    Pocn = value of phytoplankton coming in from ocean
    Zocn = value of zooplankton coming in from ocean
    Docn = value of detritus coming in from ocean
    Ts = relaxation timescale [days] for surface layer
    Td = relaxation timescale [days] for deep layer
    Ns = value of N to relax to in surface layer
    Nd = value of N to relax to in deep layer
    Ps = value of P to relax to in surface layer
    Pd = value of P to relax to in deep layer
    Zs = value of Z to relax to in surface layer
    Zd = value of Z to relax to in deep layer
    Ds = value of D to relax to in surface layer
    Dd = value of D to relax to in deep layer
    """
    
    # objects needed
    class sterms:
        def __init__(self,psterms):
            self.advright = psterms[:,:,0]
            self.reflux = psterms[:,:,1]
            self.efflux = psterms[:,:,2]
            self.outflow = psterms[:,:,3]
            self.growth = psterms[:,:,4]
            self.grazing = psterms[:,:,5]
            self.death = psterms[:,:,6]
            self.phys = self.advright + self.reflux + self.efflux + self.outflow
            self.bio = self.growth + self.grazing + self.death
    
    class dterms:
        def __init__(self,pdterms):
            self.advleft = pdterms[:,:,0]
            self.efflux = pdterms[:,:,1]
            self.reflux = pdterms[:,:,2]
            self.inflow = pdterms[:,:,3]
            self.growth = pdterms[:,:,4]
            self.grazing = pdterms[:,:,5]
            self.death = pdterms[:,:,6]
            self.phys = self.advleft + self.efflux + self.reflux + self.inflow
            self.bio = self.growth + self.grazing + self.death

    class detritus_terms:
        def __init__(self,dterms):
            self.shallow = dterms[:,:,0]
            self.deep = dterms[:,:,1]
    
    # functions needed
    def mu_i(E, N, mu0):
        # instantaneous phytoplankton growth rate
        mu0 = mu0/86400 # maximum instantaneous growth rate: s-1
        ks = 4.6 # half-satuaration for nitrate uptake: uM N
        alpha = 0.07/86400 # initial slope of growth-ligh curve: (W m-2)-1 s-1    
        mu_i = mu0 * (N/(ks + N)) * (alpha * E)/np.sqrt(mu0**2 + alpha**2*E**2)
        return mu_i

    def I(P, I0):
        # zooplankton ingestion
        I0 = I0/86400 # maximum ingestion rate: s-1
        Ks = 3 # half-satuaration for ingestion: uM N
        I = I0 * P**2 / (Ks**2 + P**2)
        return I
    
    def Ef(t): #t in days
        E0 = 200 # max light: W m-2
        E = (E0/2) * (1 + np.cos(2*np.pi*t))
        return E
    # unpack some parameters
    NS, NX, NT, dt, dvs, dvd, Qout, Qin, a0, a1, qr, way = info_tup
    # Get Q's at box edges used in the box model.
    # NOTE: all Q's are positive for the box model, whereas
    # Qout is negative in my usual TEF sign convention.
    Qout0 = Qout[:-1]
    Qin0 = Qin[:-1]
    Qout1 = Qout[1:]
    Qin1 = Qin[1:]
    '''
    Qout0 = np.zeros(NX-1)
    Qin0 = np.zeros(NX-1)
    Qout1 = np.zeros(NX-1)
    Qin1 = np.zeros(NX-1)
    '''
    NTs = int(NT/NS) # save interval, in timesteps
    # initialize arrays to save in
    nsa = np.nan * np.ones((NS,NX-1))
    nda = np.nan * np.ones((NS,NX-1))
    psa = np.nan * np.ones((NS,NX-1))
    pda = np.nan * np.ones((NS,NX-1))
    zsa = np.nan * np.ones((NS,NX-1))
    zda = np.nan * np.ones((NS,NX-1))
    dsa = np.nan * np.ones((NS,NX-1))
    dda = np.nan * np.ones((NS,NX-1))
    mu_is = np.nan * np.ones((NS,NX-1))
    mu_id = np.nan * np.ones((NS,NX-1))
    I_s = np.nan * np.ones((NS,NX-1))
    I_d = np.nan * np.ones((NS,NX-1))
    tta = 0 # index for periodic saves
    times = np.nan * np.ones((NS))
    t = np.nan * np.ones((NT))
    psterms = np.nan * np.ones((NS,NX-1,7))
    pdterms = np.nan * np.ones((NS,NX-1,7))
    detterms = np.nan * np.ones((NS,NX-1,2))
    t[0]=0
    
    #NPZD Model Biological parameters
    min_N=0.000001 #[uM N]
    min_P=0.000001 #[uM N]
    min_Z=0.000001 #[uM N]
    min_D=0.000001 #[uM N]
    
  #  m = 0.1/86400 # nongrazing mortality: s-1
    m = m/86400 # nongrazing mortality: s-1
    eps = 0.3 # growth efficiency
    xi = 2.0/86400 # mortality: s-1 (uM N)-1
    f_e = 0.5 # fraction of losses egested
    r = r/86400 # remineralization rate s-1
    attsw = 0.13 #m-1 light attenuation by seawater
    attp = 0.018 #m-1 (uM N)-1 self shading
  #  ws = 8/86400 #sinking rate m s-1 8
    seq = 0 #fraction of sinking deep detritus sequestered

    ws_array = ws*np.ones(len(dsp))

    if way == 'new':
        # force sink = 0 in the first box because the bottom cell there is not active
        ws_array[0] = 0 
    
    for tt in range(NT):
        #'''
        # boundary conditions Dirichlet
        Nout0 = np.concatenate(([Nriv], nsp[:-1])) # river
        Nin1 = np.concatenate((ndp[1:], [Nocn])) # ocean
        Pout0 = np.concatenate(([Priv], psp[:-1])) # river - no phytoplankton
        Pin1 = np.concatenate((pdp[1:], [Pocn])) # ocean - match concentration
        Zout0 = np.concatenate(([Zriv], zsp[:-1])) # river - no zooplankton
        Zin1 = np.concatenate((zdp[1:], [Zocn])) # ocean - match concentration
        Dout0 = np.concatenate(([Driv], dsp[:-1])) # river - detritus
        Din1 = np.concatenate((ddp[1:], [Docn])) # ocean - match concentration
        '''
        # boundary conditions Neumann all except N
        Nout0 = np.concatenate(([Nriv], nsp[:-1])) # river
        Nin1 = np.concatenate((ndp[1:], [Nocn])) # ocean
        Pout0 = np.concatenate(([psp[1]], psp[:-1])) # river - no phytoplankton
        Pin1 = np.concatenate((pdp[1:], [pdp[-2]])) # ocean - match concentration
        Zout0 = np.concatenate(([zsp[1]], zsp[:-1])) # river - no zooplankton
        Zin1 = np.concatenate((zdp[1:], [zdp[-2]])) # ocean - match concentration
        Dout0 = np.concatenate(([dsp[1]], dsp[:-1])) # river - detritus
        Din1 = np.concatenate((ddp[1:], [ddp[-2]])) # ocean - match concentration
        '''
        # update fields
        if tt>0:
            t[tt] = t[tt-1]+dt
        
        #Update light
        Es=Ef(t[tt]/86400) #Shallow E. Function expects it in days. 
        Ed=Es*np.exp(-attsw*hs-attp*psp*hs) #Deep E. Depends depth and shading shallow plankton population.
        
        #Nutrient
        ns = (nsp + (dt/dvs)*(Qout0*(1-a0)*Nout0 + Qin1*a1*Nin1 - Qout1*nsp + qr*Nriv)
        + dt*( -mu_i(Es, nsp, mu0)*psp + (1-eps)*(1-f_e)*I(psp,I0)*zsp + r*dsp ) + (Ns-nsp)*(dt/86400)/Ts) 
        nd = (ndp + (dt/dvd)*(Qin1*(1-a1)*Nin1 + Qout0*a0*Nout0 - Qin0*ndp ) #+ seq*ws_array*(dvd/hd)*ddp 
        + dt*( -mu_i(Ed, ndp, mu0)*pdp + (1-eps)*(1-f_e)*I(pdp,I0)*zdp + r*ddp ) + (Nd-ndp)*(dt/86400)/Td)
        ns[ns<0] = min_N
        nd[nd<0] = min_N
        if way == 'old':#old way
            nd[0] = nd[1] # this helps when using riv = ocn = const.
        else: #new way
            nd[0] = np.nan # first bottom cell not active, so mask
        
        #Phytoplankton
        ps = (psp + (dt/dvs)*(Qout0*(1-a0)*Pout0 + Qin1*a1*Pin1 - Qout1*psp + qr*Priv) 
        + dt*( mu_i(Es, nsp, mu0)*psp - I(psp,I0)*zsp - m*psp ) 
        + (Ps-psp)*(dt/86400)/Ts) 
        pd = (pdp + (dt/dvd)*(Qin1*(1-a1)*Pin1 + Qout0*a0*Pout0 - Qin0*pdp) 
        + dt*( mu_i(Ed, ndp, mu0)*pdp - I(pdp,I0)*zdp - m*pdp ) 
        + (Pd-pdp)*(dt/86400)/Td)
        ps[ps<0] = min_P
        pd[pd<0] = min_P
        if way == 'old':#old way
            pd[0] = pd[1] # this helps when using riv = ocn = const.
        else: #new way
            pd[0] = np.nan # first bottom cell not active, so mask
        
        #Zooplankton
        zs = (zsp + (dt/dvs)*(Qout0*(1-a0)*Zout0 + Qin1*a1*Zin1 - Qout1*zsp + qr*Zriv) 
        + dt*( eps*I(psp,I0)*zsp - xi*zsp**2 ) 
        + (Zs-zsp)*(dt/86400)/Ts) 
        zd = (zdp + (dt/dvd)*(Qin1*(1-a1)*Zin1 + Qout0*a0*Zout0 - Qin0*zdp) 
        + dt*( eps*I(pdp,I0)*zdp - xi*zdp**2 ) 
        + (Zd-zdp)*(dt/86400)/Td)
        zs[zs<0] = min_Z
        zd[zd<0] = min_Z
        if way == 'old':#old way
            zd[0] = zd[1] # this helps when using riv = ocn = const.
        else: #new way
            zd[0] = np.nan # first bottom cell not active, so mask
        
        #Detritus
        ds = (dsp + (dt/dvs)*(Qout0*(1-a0)*Dout0 + Qin1*a1*Din1 - Qout1*dsp + qr*Driv -ws_array*(dvs/hs)*dsp) 
        + dt*( (1-eps)*f_e*I(psp,I0)*zsp + m*psp + xi*zsp**2 - r*dsp )   
        + (Ds-dsp)*(dt/86400)/Ts) 
        dd = (ddp + (dt/dvd)*(Qin1*(1-a1)*Din1 + Qout0*a0*Dout0 - Qin0*ddp +ws_array*(dvs/hs)*dsp - seq*ws_array*(dvd/hd)*ddp) 
        + dt*( (1-eps)*f_e*I(pdp,I0)*zdp + m*pdp + xi*zdp**2 - r*ddp ) 
        + (Dd-ddp)*(dt/86400)/Td)        
        ds[ds<0] = min_D
        dd[dd<0] = min_D
        if way == 'old':#old way
            dd[0] = dd[1] # this helps when using riv = ocn = const.
        else: #new way
            dd[0] = np.nan # first bottom cell not active, so mask
              
        if (np.mod(tt, NTs) == 0) and tta < NS:
            # periodic save
            nsa[tta,:] = ns
            nda[tta,:] = nd
            psa[tta,:] = ps
            pda[tta,:] = pd
            zsa[tta,:] = zs
            zda[tta,:] = zd
            dsa[tta,:] = ds
            dda[tta,:] = dd
            mu_is[tta,:] = mu_i(Es, nsp, mu0)
            mu_id[tta,:] = mu_i(Ed, ndp, mu0)
            I_s[tta,:] = I(psp,I0)
            I_d[tta,:] = I(pdp,I0)
            times[tta] = tt
            psterms[tta,:,0] = (1/dvs)*(Qout0*(1)*Pout0) #Advection right
            psterms[tta,:,1] = (1/dvs)*(Qout0*(-a0)*Pout0) #Reflux (down) 
            psterms[tta,:,2] = (1/dvs)*(Qin1*a1*Pin1) #Efflux (up)
            psterms[tta,:,3] = (1/dvs)*(- Qout1*psp) #Outflow
            psterms[tta,:,4] = (mu_i(Es, nsp, mu0)*psp) #Growth
            psterms[tta,:,5] = (- I(psp,I0)*zsp) #Grazing
            psterms[tta,:,6] = (- m*psp) #Death
            pdterms[tta,:,0] = (1/dvd)*(Qin1*(1)*Pin1) #Advection left
            pdterms[tta,:,1] = (1/dvd)*(Qin1*(-a1)*Pin1) #Efflux (up)
            pdterms[tta,:,2] = (1/dvd)*(Qout0*a0*Pout0) #Reflux (down)
            pdterms[tta,:,3] = (1/dvd)*(- Qin0*pdp) #Inflow
            pdterms[tta,:,4] = (mu_i(Ed, ndp, mu0)*pdp) #Growth 
            pdterms[tta,:,5] = (- I(pdp,I0)*zdp) #Grazing
            pdterms[tta,:,6] = (- m*pdp) #Death 
            detterms[tta,:,0] = (1-eps)*f_e*I(psp,I0)*zsp + m*psp + xi*zsp**2 - r*dsp #Shallow D bio terms
            detterms[tta,:,1] = (1-eps)*f_e*I(pdp,I0)*zdp + m*pdp + xi*zdp**2 - r*ddp #Deep D bio terms

            tta += 1
            
        #For next run
        nsp = ns.copy()
        ndp = nd.copy()
        psp = ps.copy()
        pdp = pd.copy()
        zsp = zs.copy()
        zdp = zd.copy()
        dsp = ds.copy()
        ddp = dd.copy()
    
    psterms_obj = sterms(psterms)
    pdterms_obj = dterms(pdterms)
    dterms_obj = detritus_terms(detterms)
    
    return nsa, nda, psa, pda, zsa, zda, dsa, dda, times, psterms_obj, pdterms_obj, dterms_obj, mu_is, mu_id, I_s, I_d