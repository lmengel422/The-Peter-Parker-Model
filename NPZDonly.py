"""
Solve the efflux-reflux system with a Chatwin solution.

Parker MacCready: simplified a bit from the original for Lily Engel, UCB. 2021.03.25
Modified by Lily Engel 2021 onwards for experiments in Engel and Stacey 2024 and Engel dissertaion (added NPZD, all experiments)
NPZDonly: have ability to loop through NPZD tests.
This version saves different experiments in classes.

This runs the default "salt" experiment:
run NPZDonly 

This runs one of the other experiments:
run NPZDonly -exp Sink

"""

import numpy as np
import matplotlib.pyplot as plt

from importlib import reload
import rflx_fun_NPZDonly as rfun
reload(rfun)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-exp', type=str, default='salt')
parser.add_argument('-dec', type=float, default=0.0)
args = parser.parse_args()
exp = args.exp
dec = args.dec

#whether or not to run new boundary condition (nan in leftmost bottom cell)
way = 'new'

class NPZD_tests:
    def __init__(self,input_tup,ws,mu0,I0,mu_is,mu_id,I_s, I_d,m,r,Nriv,Nocn,csa,cda,nsa,nda,psa,pda,zsa,zda,dsa,dda,times,psterms,pdterms,seq,dterms):
        Socn,Qin,Qout,B,Qr,Sin,Sout,hs,hd,L,a0,a1,dt,NT,NS,xm,ds,dvs,dvd = input_tup
        self.Socn = Socn
        self.Qout = Qout[-1]
        self.Qout_full = Qout
        self.Qin_full = Qin
        self.ws = ws
        self.ws_day = ws*86400
        self.B = B
        self.Qr = Qr[-1]
        self.mu0 = mu0
        self.I0 = I0
        self.mu_is = mu_is
        self.mu_id = mu_id
        self.I_s = I_s
        self.I_d = I_d
        self.m = m
        self.r_day = r # remineralization rate d-1
        self.r = r/86400 # remineralization rate s-1
        self.Nriv = Nriv
        self.Nocn = Nocn
        self.Sin = Sin
        self.Sout = Sout
        self.L = L
        self.L_real = x[-1]-x[0]
        self.hs = hs
        self.hd = hd
        self.a0 = a0
        self.a1 = a1
        self.dt = dt
        self.NT = NT
        self.NS = NS
        self.ds = ds
        self.seq = seq
        self.dvs = dvs
        self.dvd = dvd
        self.Vs = self.B*self.L_real*self.hs
        self.Vd = self.B*self.L_real*self.hd
        self.csa = csa
        self.cda = cda
        self.nsa = nsa
        self.nda = nda
        self.psa = psa
        self.pda = pda
        self.zsa = zsa
        self.zda = zda
        self.dsa = dsa
        self.dda = dda
        self.Xm = xm/1000
        self.times = times
        self.psterms = psterms
        self.pdterms = pdterms
        self.dim_B = self.B/self.L_real
        self.dim_Qr = self.Qout/self.Qr
        self.dim_mu0 = self.mu0*self.L_real*self.L_real*self.hs/self.Qr
        self.dim_I0 = self.I0*self.L_real*self.L_real*self.hs/self.Qr
        self.dim_m = self.m*self.L_real*self.L_real*self.hs/self.Qr
        self.dim_sink = self.ws*self.L_real*self.L_real/self.Qr
        if way == 'new':
            self.maxpd = np.max(pda[-1,1:])
            self.locpd = np.argmax(pda[-1,1:])
            self.maxdd = np.max(dda[-1,1:])
            self.locdd = np.argmax(dda[-1,1:])
        else:
            self.maxpd = np.max(pda[-1,:])
            self.locpd = np.argmax(pda[-1,:])
            self.maxdd = np.max(dda[-1,:])
            self.locdd = np.argmax(dda[-1,:])
        self.maxps = np.max(psa[-1,:])
        self.locps = np.argmax(psa[-1,:])
        self.maxds = np.max(dsa[-1,:])
        self.locds = np.argmax(dsa[-1,:])
        if self.locpd == 98:
            self.locpd = 97
        self.ints = np.trapz(psa[-1,:],self.Xm)
        self.intd = np.trapz(pda[-1,1:],self.Xm[1:])
        self.ints3d = np.trapz(psa[-1,:],self.Xm)*self.B
        self.intd3d = np.trapz(pda[-1,1:],self.Xm[1:])*self.B
        self.int = self.ints/(self.ints+self.intd)
        self.intDs = np.trapz(dsa[-1,:],self.Xm)
        self.intDd = np.trapz(dda[-1,1:],self.Xm[1:])
        self.intsD3d = np.trapz(dsa[-1,:],self.Xm)*self.B
        self.intdD3d = np.trapz(dda[-1,1:],self.Xm[1:])*self.B
        self.intD = self.intDs/(self.intDs+self.intDd)
        self.int_totDd = self.intDd/(self.intDs+self.intDd)
       # self.Nriv = self.Nriv/self.Nocn #FIXME - divide by 0
        self.Qout0 = Qout[:-1]
        self.Qin0 = Qin[:-1]
        self.Qout1 = Qout[1:]
        self.Qin1 = Qin[1:]
        #NOTE - USING DETRITUS MAX HERE ------------------------------
        if self.ws > 0:
            self.tau_sink_shall_P = self.hs*self.psa[-1,:]/(self.ws*self.dsa[-1,:])
            self.tau_sink_deep_P = self.hd*self.pda[-1,:]/(self.ws*self.dda[-1,:])
            self.tau_sink_shall = self.hs/(self.ws)
            self.tau_sink_deep = self.hd/(self.ws)
        else:
            self.tau_sink_shall = 100*86400*np.ones(NX-1)
            self.tau_sink_deep = 100*86400*np.ones(NX-1)
        self.tau_q_shall = self.Vs/self.Qout
        if self.Qout_full[self.locds] > 0:
            self.tau_q_shall_v2 = xm[self.locds]*self.B*self.hs/self.Qout_full[self.locds]
            self.tau_q_shall_v4 = self.dvs[self.locds]/(self.Qout_full[self.locds]) #USE THIS ONE
        else:
            self.tau_q_shall_v2 = 100*86400
            self.tau_q_shall_v4 = 100*86400
        self.tau_q_deep = self.Vd/self.Qout
        if self.Qin_full[self.locdd] > 0:
            self.tau_q_deep_v2 = (xm[-1]-xm[self.locdd])*self.B*self.hd/self.Qin_full[self.locdd]
            self.tau_q_deep_v4 = self.dvd[self.locdd]/(self.Qin_full[self.locdd]) #USE THIS ONE
        else:
            self.tau_q_deep_v2 = 100*86400
            self.tau_q_deep_v4 = 100*86400
        self.net_q_up = np.abs(self.Qout0*(-self.a0)+self.Qin1*self.a1)
        self.net_q_down = np.abs(self.Qout0*(self.a0)-self.Qin1*self.a1)
        self.tau_net_up = self.dvs/(self.net_q_up)
        self.tau_net_down = self.dvd/np.abs(self.net_q_down)
        self.Kdisp = ((1-self.a0)*self.Qout0+(1-a1)*self.Qin1)**2*self.L_real/((self.Qout0*self.a0+self.Qin1*self.a1)*self.B*(self.hs+self.hd))
        self.tau_disp = self.L_real**2/self.Kdisp[1:]
        self.tau_grow = 1/self.mu_is[-1,:]
        self.tau_graze = psa[-1,:]/(self.I_s[-1,:]*zsa[-1,:]) #s
        self.tau_grow_d = 1/self.mu_id[-1,:] #s
        self.tau_graze_d = pda[-1,:]/(self.I_d[-1,:]*zda[-1,:]) #s
        self.tau_die = 1/self.m #d
        self.eps = 0.3 # growth efficiency
        self.xi = 2.0/86400 # mortality: s-1 (uM N)-1
        self.f_e = 0.5 # fraction of losses egested
        self.tau_messy = 1/((1-self.eps)*self.f_e*self.I_s[-1,:]) #s
        self.tau_messy_d = 1/((1-self.eps)*self.f_e*self.I_d[-1,:]) #s 
        self.tau_zdie = zsa[-1,:]/self.xi #s
        self.tau_zdie_d = zda[-1,:]/self.xi #s
        self.tau_remin = 1/self.r #s
        self.dterms = dterms
    #    self.tau_phys_s = 1/(1/self.tau_q_shall_v2 + 1/self.tau_net_up[self.locps] + 1/self.tau_disp[self.locps])
     #   self.tau_phys_d = 1/(1/self.tau_q_deep_v2 + 1/self.tau_net_down[self.locpd] + 1/self.tau_disp[self.locpd])
    #    self.tau_bio_s = 1/(1/self.tau_grow[self.locps] + 1/self.tau_graze[self.locps] + 1/(self.tau_die*86400))
    #    self.tau_bio_d = 1/(1/self.tau_grow_d[self.locpd] + 1/self.tau_graze_d[self.locpd] + 1/(self.tau_die*86400))
        
    def __str__(self):
        return f"ws = {self.ws_day} m/d, Qout = {self.Qout} m^3/s, B = {self.B} m, Qr = {self.Qr} m^3/s, mu0 = {self.mu0} 1/d, I0 = {self.I0} 1/d, m = {self.m} 1/d, Nriv = {self.Nriv} uM N"

if exp == 'change_Socn':
    Socn_range = [5, 10, 15, 20, 25, 30, 35, 40]
     #   Socn_range = np.linspace(5,100,20)
elif exp == 'change_Socn_zoom':
    Socn_range = [5, 6, 7, 8, 9, 10]
else:
    Socn_range = [30] #ocean salinity 30

L = 50e3
#q_range = np.nan * np.ones(len(Socn_range))
#q_range = 200*(np.array(Socn_range)-5)+1500 #Precalculate q's to make other test variables- want to be ascending. saveq variable later is what actual q's were to compare.
q_range = np.nan * np.ones(20)
q_range = np.array(np.linspace(1500, 20500, 20))
testq = 0.00025*3e3*L/q_range

if exp == 'change_B':
    B_range = [3e2, 5e2, 1e3, 3e3, 4e3, 5e3, 7e3, 10e3, 3e4, 3e5]
 #   B_range =q_range*6500/(0.00025*L)
else:
    B_range = [3e3] #width (m)
    
if exp == 'change_qr':
    qr_range = [250,500,1000,2000,3000,4000,5000,6000,7000]
else:
    qr_range = [1000] #river flux (m/s)

for ii in range(len(B_range)):
    for jj in range(len(Socn_range)):
        for kk in range(len(qr_range)):
            # Naming conventions
            #  Layers: s = shallow, d = deep

            # define bathymetry and layer thickness
            B = B_range[ii]    # width (m)
            hs = 20     # thickness of shallow layer (m)
            hd = 20     # thickness of deep layer (m)
            nx = 100    # number of steps to initially define a channel
            # estuary physical parameters
            Socn = Socn_range[jj]  # ocean salinity 
            ds = 5 # Sin - Sout at the mouth

            qr = np.zeros(nx-1)
            qr[0] = qr_range[kk]
            if exp == 'qr1': #FIXME - combine qr1 and qr2 to be able to have multiple tributaries?
                qr[int(nx/3)] = qr_range[kk]
            elif exp == 'qr2':
                qr[int(nx/3)] = qr_range[kk]
                qr[int(2*nx/3)] = qr_range[kk]
            Qr = np.cumsum(qr) # river flow [m3/s]
            Qr = np.concatenate(([0], Qr))

            # get the solution at box edges
            Sin, Sout, x, L = rfun.get_Sio_chatwin(Socn, ds, nx)
            #Sin, Sout, x, L = rfun.get_Sio_fjord(Socn, ds, nx)
            Qout = Qr*Sin/(Sin - Sout) #Knudsen 
            Qin = Qr*Sout/(Sin - Sout)

            # run specifications
            ndays = 200 
            dx = np.diff(x)
            NX = len(x)
            xm = x[:-1] + dx/2 # x at box centers
            # box volumes (at box centers)
            dvs = B*hs*dx
            dvd = B*hd*dx
            # calculate Efflux-Reflux fractions (a11 and a00) for each box
            a0, a1 = rfun.a_calc(Sin, Sout)
            # get time step
            dt, NT, NS = rfun.get_time_step(dx, Qout, B, hs, ndays)
            # pack some parameters
            info_tup = (NS, NX, NT, dt, dvs, dvd, Qout, Qin, a0, a1, qr, way)
            input_tup = (Socn,Qin,Qout,B,Qr,Sin,Sout,hs,hd,L,a0,a1,dt,NT,NS,xm,ds,dvs,dvd)

            # intial condition vectors
            csp = np.zeros(NX-1) # csp = "concentration shallow previous"
            cdp = np.zeros(NX-1) # cdp = "concentration deep previous"
            nsp = np.zeros(NX-1) # nsp = "concentration N shallow previous"
            ndp = np.zeros(NX-1) # ndp = "concentration N deep previous"
            psp = np.zeros(NX-1) # psp = "concentration P shallow previous"
            pdp = np.zeros(NX-1) # pdp = "concentration P deep previous"
            zsp = np.zeros(NX-1) # zsp = "concentration Z shallow previous"
            zdp = np.zeros(NX-1) # zdp = "concentration Z deep previous"
            dsp = np.zeros(NX-1) # dsp = "concentration D shallow previous"
            ddp = np.zeros(NX-1) # ddp = "concentration D deep previous"

            #NPZD population and reproduce salinity state
            N0=5 #[uM N] 30 
            P0=0.01 #[uM N] 1
            Z0=0.01 #[uM N] 0.1
            D0=0  #[uM N] 0
            Nocn = 0 #rfun.nut_sal(Socn)  #[uM N] 
            
            #Default values for non-test cases
            mu0 = 2.2 # max inst growth rate: d-1 2.2
            I0 = 4.8 #max grazing rate d-1
            ws = 8/86400 #d sinking rate m s-1 8
            m = 0.1 #mortality
            r = 0.1 #remineralization 
            Nriv = N0
            seq = 0

            plot_s = False
            if exp == 'salt':
                # Reproduce salinity state
                csa, cda, times = rfun.c_calc(csp, cdp, info_tup, ocn=Sin[-1])
                nsa, nda, psa, pda, zsa, zda, dsa, dda, times, psterms, pdterms, dterms, mu_is, mu_id, I_s, I_d = rfun.npzd_calc(nsp,ndp,psp,pdp,zsp,zdp,dsp,ddp,hs,hd,info_tup,Nriv=Nriv,Nocn=Nocn,Priv=P0,Pocn=P0,Zriv=Z0,Zocn=Z0)

                salt_NPZD = NPZD_tests(input_tup,ws,mu0,I0,mu_is,mu_id,I_s, I_d,m,r,Nriv,Nocn,csa,cda,nsa,nda,psa,pda,zsa,zda,dsa,dda,times,psterms,pdterms,seq,dterms)

            elif exp == 'D_Sink':
            #    ws_range = np.array([0, 0.00001, 0.00005, 8/86400, 0.00025, 0.0005, 0.001])
                ws_range = testq*6500/(B*L)
                ws_range = np.flip(ws_range)
                
                D_Sink = {}
                for i in range(len(ws_range)):
                    ws = ws_range[i]
                    riv = 1
                    csa, cda, times = rfun.c_calc(csp, cdp, info_tup, ocn=Sin[-1])
                    nsa, nda, psa, pda, zsa, zda, dsa, dda, times, psterms, pdterms, dterms, mu_is, mu_id, I_s, I_d = rfun.npzd_calc_change_ecol(nsp,ndp,psp,pdp,zsp,zdp,dsp,ddp,hs,hd,info_tup,ws=ws,Nriv=N0,Nocn=Nocn,Priv=P0,Pocn=P0,Zriv=Z0,Zocn=Z0)

                    D_Sink['ws{0}'.format(i)] = NPZD_tests(input_tup,ws,mu0,I0,mu_is,mu_id,I_s, I_d,m,r,Nriv,Nocn,csa,cda,nsa,nda,psa,pda,zsa,zda,dsa,dda,times,psterms,pdterms,seq,dterms)

            elif exp == 'change_B':
                if ii==0:
                    change_B_NPZD = {}

                csa, cda, times = csa, cda, times = rfun.c_calc(csp, cdp, info_tup, ocn=Sin[-1])
                nsa, nda, psa, pda, zsa, zda, dsa, dda, times, psterms, pdterms, dterms, mu_is, mu_id, I_s, I_d = rfun.npzd_calc(nsp,ndp,psp,pdp,zsp,zdp,dsp,ddp,hs,hd,info_tup,Nriv=N0,Nocn=Nocn,Priv=P0,Pocn=P0,Zriv=Z0,Zocn=Z0)
                
                change_B_NPZD['B{0}'.format(ii)] = NPZD_tests(input_tup,ws,mu0,I0,mu_is,mu_id,I_s, I_d,m,r,Nriv,Nocn,csa,cda,nsa,nda,psa,pda,zsa,zda,dsa,dda,times,psterms,pdterms,seq,dterms)

            elif exp == 'change_Socn' or exp == 'change_Socn_zoom':
                if jj == 0 and exp == 'change_Socn':
                    change_Socn_NPZD = {}
                elif jj == 0 and exp == 'change_Socn_zoom':
                    change_Socn_zoom_NPZD = {}

                csa, cda, times = csa, cda, times = rfun.c_calc(csp, cdp, info_tup, ocn=Sin[-1])
                nsa, nda, psa, pda, zsa, zda, dsa, dda, times, psterms, pdterms, dterms, mu_is, mu_id, I_s, I_d = rfun.npzd_calc(nsp,ndp,psp,pdp,zsp,zdp,dsp,ddp,hs,hd,info_tup,Nriv=N0,Nocn=Nocn,Priv=P0,Pocn=P0,Zriv=Z0,Zocn=Z0)

                if exp == 'change_Socn':
                    change_Socn_NPZD['Qout{0}'.format(jj)] = NPZD_tests(input_tup,ws,mu0,I0,mu_is,mu_id,I_s, I_d,m,r,Nriv,Nocn,csa,cda,nsa,nda,psa,pda,zsa,zda,dsa,dda,times,psterms,pdterms,seq,dterms)
                else:
                    change_Socn_zoom_NPZD['Qout{0}'.format(jj)] = NPZD_tests(input_tup,ws,mu0,I0,mu_is,mu_id,I_s, I_d,m,r,Nriv,Nocn,csa,cda,nsa,nda,psa,pda,zsa,zda,dsa,dda,times,psterms,pdterms,seq,dterms)

         #       print('Water flux out at mouth %0.3f [c m3/s]' % (-Qout[-1]))
        #        print('Water flux in at mouth %0.3f [c m3/s]' % (Qin[-1]))
         #       print('Water flux in at head %0.3f [c m3/s]' % (Qr[-1]))

            elif exp == 'change_qr':
                if kk==0:
                    change_qr = {}

                csa, cda, times = csa, cda, times = rfun.c_calc(csp, cdp, info_tup, ocn=Sin[-1])
                nsa, nda, psa, pda, zsa, zda, dsa, dda, times, psterms, pdterms, dterms, mu_is, mu_id, I_s, I_d = rfun.npzd_calc(nsp,ndp,psp,pdp,zsp,zdp,dsp,ddp,hs,hd,info_tup,Nriv=N0,Nocn=Nocn,Priv=P0,Pocn=P0,Zriv=Z0,Zocn=Z0)
                
                change_qr['Qr{0}'.format(kk)] = NPZD_tests(input_tup,ws,mu0,I0,mu_is,mu_id,I_s, I_d,m,r,Nriv,Nocn,csa,cda,nsa,nda,psa,pda,zsa,zda,dsa,dda,times,psterms,pdterms,seq,dterms)
                
            #    print('Water flux out at mouth %0.3f [c m3/s]' % (-Qout[-1]))
           #     print('Water flux in at mouth %0.3f [c m3/s]' % (Qin[-1]))
            #    print('Water flux in at head %0.3f [c m3/s]' % (Qr[-1]))

            elif exp == 'change_Nriv':
                Nriv_range = [5,10,15,20,30,40]

                change_Nriv = {}
                for i in range(len(Nriv_range)):
                    Nriv = Nriv_range[i]
                    csa, cda, times = rfun.c_calc(csp, cdp, info_tup, ocn=Sin[-1])
                    nsa, nda, psa, pda, zsa, zda, dsa, dda, times, psterms, pdterms, dterms, mu_is, mu_id, I_s, I_d = rfun.npzd_calc(nsp,ndp,psp,pdp,zsp,zdp,dsp,ddp,hs,hd,info_tup,Nriv=Nriv,Nocn=Nocn,Priv=P0,Pocn=P0,Zriv=Z0,Zocn=Z0)

                    change_Nriv['Nriv{0}'.format(i)] = NPZD_tests(input_tup,ws,mu0,I0,mu_is,mu_id,I_s, I_d,m,r,Nriv,Nocn,csa,cda,nsa,nda,psa,pda,zsa,zda,dsa,dda,times,psterms,pdterms,seq,dterms)
            
            elif exp == 'P_growth':
                mu0_range = [1.1,2.2,4.4,8.8,17.6] # max inst growth rate: d-1
                P_growth = {}

                for i in range(len(mu0_range)):
                    mu0 = mu0_range[i]
                    riv = 1
                    csa, cda, times = rfun.c_calc(csp, cdp, info_tup, ocn=Sin[-1])
                    nsa, nda, psa, pda, zsa, zda, dsa, dda, times, psterms, pdterms, dterms, mu_is, mu_id, I_s, I_d = rfun.npzd_calc_change_ecol(nsp,ndp,psp,pdp,zsp,zdp,dsp,ddp,hs,hd,info_tup,ws,Nriv=N0,Nocn=Nocn,Priv=P0,Pocn=P0,Zriv=Z0,Zocn=Z0,mu0=mu0,I0=I0,m=m)

                    P_growth['mu0{0}'.format(i)] = NPZD_tests(input_tup,ws,mu0,I0,mu_is,mu_id,I_s, I_d,m,r,Nriv,Nocn,csa,cda,nsa,nda,psa,pda,zsa,zda,dsa,dda,times,psterms,pdterms,seq,dterms)
        
            elif exp == 'Z_graze':
                I0_range = [2.4,4.8,9.6,19.2,38.4] # maximum ingestion rate: d-1
                Z_graze = {}

                for i in range(len(I0_range)):
                    I0 = I0_range[i]
                    riv = 1
                    csa, cda, times = rfun.c_calc(csp, cdp, info_tup, ocn=Sin[-1])
                    nsa, nda, psa, pda, zsa, zda, dsa, dda, times, psterms, pdterms, dterms, mu_is, mu_id, I_s, I_d = rfun.npzd_calc_change_ecol(nsp,ndp,psp,pdp,zsp,zdp,dsp,ddp,hs,hd,info_tup,ws,Nriv=N0,Nocn=Nocn,Priv=P0,Pocn=P0,Zriv=Z0,Zocn=Z0,mu0=mu0,I0=I0,m=m)

                    Z_graze['I0{0}'.format(i)] = NPZD_tests(input_tup,ws,mu0,I0,mu_is,mu_id,I_s, I_d,m,r,Nriv,Nocn,csa,cda,nsa,nda,psa,pda,zsa,zda,dsa,dda,times,psterms,pdterms,seq,dterms)
            
            elif exp == 'P_mort':
                m_range = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8] # nongrazing mort: d-1
                
                P_mort = {}

                for i in range(len(m_range)):
                    m = m_range[i]
                    riv = 1
                    csa, cda, times = rfun.c_calc(csp, cdp, info_tup, ocn=Sin[-1])
                    nsa, nda, psa, pda, zsa, zda, dsa, dda, times, psterms, pdterms, dterms, mu_is, mu_id, I_s, I_d = rfun.npzd_calc_change_ecol(nsp,ndp,psp,pdp,zsp,zdp,dsp,ddp,hs,hd,info_tup,ws,Nriv=N0,Nocn=Nocn,Priv=P0,Pocn=P0,Zriv=Z0,Zocn=Z0,mu0=mu0,I0=I0,m=m)

                    P_mort['m{0}'.format(i)] = NPZD_tests(input_tup,ws,mu0,I0,mu_is,mu_id,I_s, I_d,m,r,Nriv,Nocn,csa,cda,nsa,nda,psa,pda,zsa,zda,dsa,dda,times,psterms,pdterms,seq,dterms)
                    
            elif exp == 'change_Nocn':
                Nocn_range = [5,10,15,20,30,40]

                change_Nocn = {}
                for i in range(len(Nocn_range)):
                    Nocn = Nocn_range[i]
                    csa, cda, times = rfun.c_calc(csp, cdp, info_tup, ocn=Sin[-1])
                    nsa, nda, psa, pda, zsa, zda, dsa, dda, times, psterms, pdterms, dterms, mu_is, mu_id, I_s, I_d = rfun.npzd_calc(nsp,ndp,psp,pdp,zsp,zdp,dsp,ddp,hs,hd,info_tup,Nriv=Nriv,Nocn=Nocn,Priv=P0,Pocn=P0,Zriv=Z0,Zocn=Z0)

                    change_Nocn['Nocn{0}'.format(i)] = NPZD_tests(input_tup,ws,mu0,I0,mu_is,mu_id,I_s, I_d,m,r,Nriv,Nocn,csa,cda,nsa,nda,psa,pda,zsa,zda,dsa,dda,times,psterms,pdterms,seq,dterms)
            
            elif exp == 'change_remin':
                r_range = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8] # remineralization: d-1
                
                change_remin = {}

                for i in range(len(r_range)):
                    r = r_range[i]
                    riv = 1
                    csa, cda, times = rfun.c_calc(csp, cdp, info_tup, ocn=Sin[-1])
                    nsa, nda, psa, pda, zsa, zda, dsa, dda, times, psterms, pdterms, dterms, mu_is, mu_id, I_s, I_d = rfun.npzd_calc_change_ecol(nsp,ndp,psp,pdp,zsp,zdp,dsp,ddp,hs,hd,info_tup,ws,Nriv=N0,Nocn=Nocn,Priv=P0,Pocn=P0,Zriv=Z0,Zocn=Z0,mu0=mu0,I0=I0,m=m,r=r)

                    change_remin['r{0}'.format(i)] = NPZD_tests(input_tup,ws,mu0,I0,mu_is,mu_id,I_s, I_d,m,r,Nriv,Nocn,csa,cda,nsa,nda,psa,pda,zsa,zda,dsa,dda,times,psterms,pdterms,seq,dterms)
            else:
                print('exp = %s not supported' % (exp))
        #        sys.exit()
            X = x/1e3
            Xm = xm/1e3

"""
# PLOTTING
plt.close('all')

fs = 16
lw = 3
plt.rc('font', size=fs)
fig = plt.figure(figsize=(12,10))
cin = 'r'
cout = 'b'
c0 = 'g'
c1 = 'darkorange'
c2 = 'm'
c3 = 'c'

ax = fig.add_subplot(411)
# add initial and final model state
ax.plot(Xm, csa[-1,:], '-', color=cout, lw=lw)
ax.plot(Xm, cda[-1,:], '-', color=cin, lw=lw) 
if plot_s:
    ax.plot(X, Sin, ':r', X, Sout, ':b', lw=lw)
    ax.text(.1, .6, 'Dotted = Target Solution', transform=ax.transAxes)
    ax.text(.1, .5, 'Solid = Numerical Solution', transform=ax.transAxes)
ax.grid(True)
#ax.set_xlim(0,X[-1])
ax.set_ylabel('Tracer')   
ax.text(.2, .8, r'$Deep$', color=cin, transform=ax.transAxes, size=1.5*fs,
    bbox=dict(facecolor='w', edgecolor='None', alpha=0.5))
ax.text(.4, .8, r'$Shallow$', color=cout, transform=ax.transAxes, size=1.5*fs,
    bbox=dict(facecolor='w', edgecolor='None', alpha=0.5))
#ax.set_ylim(0, 1.1*np.max([cda[-1,:].max(),csa[-1,:].max()]))
ax.set_xlabel('X (km)')
ax.set_title(exp.replace('_',' '))

ax = fig.add_subplot(412)
ax.plot(Xm, a0, '-', color=c0, lw=lw)
ax.plot(Xm, a1, '-', color=c1, lw=lw)
ax.grid(True)
ax.set_xlabel('X (km)')
ax.set_ylabel('Fraction')
ax.text(.1, .85, r'Reflux $a_{0}$', color=c0, transform=ax.transAxes, size=1.3*fs,
    bbox=dict(facecolor='w', edgecolor='None', alpha=0.5))
ax.text(.1, .7, r'Efflux $a_{1}$', color=c1, transform=ax.transAxes, size=1.3*fs,
    bbox=dict(facecolor='w', edgecolor='None', alpha=0.5))
#ax.set_xlim(0,X[-1])
ax.set_ylim(0, 1.1)
"""
