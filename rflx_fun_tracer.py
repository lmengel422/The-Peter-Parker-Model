"""
Functions for the rflx code.
"""

import numpy as np

# function to create Sin and Sout
def get_Sio_chatwin(Socn, ds, nx, L=50e3):
    a = Socn/(L**1.5)
    alpha = ds/L
    x = np.linspace((alpha/(2*a))**2,L,nx)
 #   x = np.linspace(2.6832815729997478e-06,L,nx)
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
    NS = 10 # number of saves
    return dt, NT, NS

def c_calc(csp, cdp, info_tup, riv=0, ocn=0, Ts=np.inf, Td=np.inf, Cs=1, Cd=0):
    """
    This is the main computational engine for the original time dependent solution.
    
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


def t_calc(csp, cdp, hs, hd, info_tup, dec, ws=0, seq=0, riv=0, ocn=0, Ts=np.inf, Td=np.inf, Cs=1, Cd=0):
    """
    This is the main computational engine for the time dependent solution. This is the modified function for sinking tracer tests.
    
    csp, cdp = vectors [xm] of initial tracer concentrations in the two layers
    csa, cda = arrays [time, xm] of the surface and deep concentrations
        over the course of the simulation
    
    info_tup = tuple of properties defining the grid, timestep and circulation
    
    dec = decay rate of sinking particle
    ws = sinking rate
    riv = value of tracer coming in from river
    ocn = value of tracer coming in from ocean
    Ts = relaxation timescale [days] for surface layer
    Td = relaxation timescale [days] for deep layer
    Cs = value of tracer to relax to in surface layer
    Cd = value of tracer to relax to in deep layer
    seq = fraction of particles to sink out of deep layer
    """
    # objects needed
    class sterms:
        def __init__(self,csterms):
            self.advright = csterms[:,:,0]
            self.reflux = csterms[:,:,1]
            self.efflux = csterms[:,:,2]
            self.outflow = csterms[:,:,3]
            self.sink = csterms[:,:,4]
            self.no_sink = self.advright + self.efflux + self.reflux + self.outflow
    
    class dterms:
        def __init__(self,cdterms):
            self.advleft = cdterms[:,:,0]
            self.efflux = cdterms[:,:,1]
            self.reflux = cdterms[:,:,2]
            self.inflow = cdterms[:,:,3]
            self.sink = cdterms[:,:,4]
            self.no_sink = self.advleft + self.efflux + self.reflux + self.inflow
    
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
    csterms = np.nan * np.ones((NS,NX-1,7))
    cdterms = np.nan * np.ones((NS,NX-1,7))
    dec = dec/86400 #decay rate s-1

    #make ws a whole array
    ws_array = ws*np.ones(len(csp))

    if way == 'new':
        # force sink = 0 in the first box because the bottom cell there is not active
        ws_array[0] = 0 
    
    for tt in range(NT):
        # boundary conditions
        Cout0 = np.concatenate(([riv], csp[:-1])) # river
        Cin1 = np.concatenate((cdp[1:], [ocn])) # ocean
        # update fields
        cs = csp + (dt/dvs)*(Qout0*(1-a0)*Cout0 + Qin1*a1*Cin1 - Qout1*csp + qr*riv -ws_array*(dvs/hs)*csp ) + dt*(dec*csp) + (Cs-csp)*(dt/86400)/Ts 
        cd = cdp + (dt/dvd)*( Qin1*(1-a1)*Cin1 + Qout0*a0*Cout0 - Qin0*cdp +ws_array*(dvs/hs)*csp - seq*ws_array*(dvd/hd)*cdp )  + dt*(-dec*cdp)  + (Cd-cdp)*(dt/86400)/Td
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
            csterms[tta,:,0] = (dt/dvs)*(Qout0*(1)*Cout0) #Advection right
            csterms[tta,:,1] = (dt/dvs)*(Qout0*(-a0)*Cout0) #Reflux (down)
            csterms[tta,:,2] = (dt/dvs)*(Qin1*a1*Cin1) #Efflux (up)
            csterms[tta,:,3] = (dt/dvs)*(- Qout1*csp) #Outflow
            csterms[tta,:,4] = (dt/dvs)*(-ws_array*(dvs/hs)*csp ) #Sink shal-deep
            cdterms[tta,:,0] = (dt/dvd)*(Qin1*(1)*Cin1) #Advection left
            cdterms[tta,:,1] = (dt/dvd)*(Qin1*(-a1)*Cin1) #Efflux (up)
            cdterms[tta,:,2] = (dt/dvd)*(Qout0*a0*Cout0) #Reflux (down)
            cdterms[tta,:,3] = (dt/dvd)*(- Qin0*cdp) #Inflow
            cdterms[tta,:,4] = (dt/dvd)*(- seq*ws_array*(dvd/hd)*cdp) #Sink deep-bot
            tta += 1
            
            #FIXME - add riverflow
            
        csterms_obj = sterms(csterms)
        cdterms_obj = dterms(cdterms)
        
    return csa, cda, times, csterms_obj, cdterms_obj