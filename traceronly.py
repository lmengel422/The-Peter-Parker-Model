"""
Solve the efflux-reflux system with a Chatwin solution.

Simplified a bit from the original for Lily Engel, UCB. 2021.03.25
Modified by Lily Engel 2021 onwards for experiments in Engel and Stacey 2024 and Engel dissertaion (added sinking, all experiments)
traceronly: have ability to loop through tracer tests.
This version saves different experiments in classes.

This runs the default "salt" experiment:
run traceronly 

This runs one of the other experiments:
run traceronly -exp TracSink

"""


import numpy as np
import matplotlib.pyplot as plt

from importlib import reload
import rflx_fun_tracer as rfun
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

#Create class to save variables for each test.
class tracer_tests:
    def __init__(self,input_tup,ws,csa,cda,times,csterms,cdterms,seq,dec):
        Socn,Qin,Qout,B,Qr,Sin,Sout,L,hs,hd,a0,a1,dt,NT,NS,xm,x,ds,dvs,dvd = input_tup
        self.Socn = Socn
        self.Qout = Qout[-1]
        self.Qout_full = Qout
        self.Qin_full = Qin
        self.ws = ws
        self.ws_day = ws*86400
        self.B = B
        self.Qr = Qr[-1]
        self.Sin = Sin
        self.Sout = Sout
        self.L = L
        self.hs = hs
        self.hd = hd
        self.a0 = a0
        self.a1 = a1
        self.dt = dt
        self.NT = NT
        self.NS = NS
        self.csa = csa
        self.cda = cda
        self.dvs = dvs
        self.dvd = dvd
        self.dec = dec
        self.L_real = x[-1]-x[0]
        self.Vs = self.B*self.L_real*self.hs
        self.Vd = self.B*self.L_real*self.hd
        self.dim1 = (ws*self.L_real*self.L_real)/(self.Qr) 
     #   self.dim1 = (ws*self.L_real*self.B)/(self.Qout) 
        self.dim2 = (Qout[-1])/(Qr[-1])
        self.dim2v2 = (Qr[-1])/(Qout[-1])
        self.dim3 = B/self.L_real
        self.Xm = xm/1000
        self.X = x/1000
        self.times = times
        self.maxs = np.max(csa[-1,:])
        self.locs = np.argmax(csa[-1,:])
        if way == 'new':
            self.maxd = np.max(cda[-1,1:])
            self.locd = np.argmax(cda[-1,1:])
        else:
            self.maxd = np.max(cda[-1,:])
            self.locd = np.argmax(cda[-1,:])
        self.ints = np.trapz(csa[-1,:],self.Xm)
        self.intd = np.trapz(cda[-1,1:],self.Xm[1:])
        self.int = self.ints/(self.ints+self.intd)
        self.int_totd = self.intd/(self.ints+self.intd)
        self.ds = ds
        self.dim4 = self.ds/(self.Socn)
        self.seq = seq
        self.seq_per = self.seq*100
        self.dim5v1 = self.dim1*seq
        self.dim5v2 = self.dim1*(1-seq) #don't use
        self.csterms = csterms
        self.cdterms = cdterms
        self.Qout0 = Qout[:-1]
        self.Qin0 = Qin[:-1]
        self.Qout1 = Qout[1:]
        self.Qin1 = Qin[1:]
        if self.ws > 0:
            self.tau_sink_shall = self.hs/self.ws
            self.tau_sink_deep = self.hd/self.ws
        else:
            self.tau_sink_shall = 100*86400
            self.tau_sink_deep = 100*86400
        self.tau_q_shall = self.Vs/self.Qout
        if self.Qout_full[self.locs] > 0:
            self.tau_q_shall_v2 = xm[self.locs]*self.B*self.hs/self.Qout_full[self.locs] 
            self.tau_q_shall_v3 = xm[self.locs]*self.B*self.hs/((1-self.a0[self.locs])*self.Qout0[self.locs])
            self.tau_q_shall_v4 = self.dvs[self.locs]/(self.Qout_full[self.locs]) #USE THIS ONE
        else:
            self.tau_q_shall_v2 = 100*86400
            self.tau_q_shall_v3 = 100*86400
            self.tau_q_shall_v4 = 100*86400
        self.tau_q_deep = self.Vd/self.Qout
        if self.Qin_full[self.locd] > 0:
            self.tau_q_deep_v2 = (xm[-1]-xm[self.locd])*self.B*self.hd/self.Qin_full[self.locd] 
            self.tau_q_deep_v3 = xm[self.locd]*self.B*self.hd/((1-self.a1[self.locd])*self.Qin1[self.locd])
            self.tau_q_deep_v4 = self.dvd[self.locd]/(self.Qin_full[self.locd]) #USE THIS ONE
        else:
            self.tau_q_deep_v2 = 100*86400
            self.tau_q_deep_v3 = 100*86400
            self.tau_q_deep_v4 = 100*86400
        self.net_q_up = np.abs(self.Qout0*(-self.a0)+self.Qin1*self.a1)
        self.net_q_down = np.abs(self.Qout0*(self.a0)-self.Qin1*self.a1)
        self.tau_net_up = self.dvs/(self.net_q_up)
        self.tau_net_down = self.dvd/np.abs(self.net_q_down)
        self.Kdisp = ((1-self.a0)*self.Qout0+(1-a1)*self.Qin1)**2*self.L_real/((self.Qout0*self.a0+self.Qin1*self.a1)*self.B*(self.hs+self.hd))
        self.tau_disp = self.L_real**2/(1*self.Kdisp[1:]) #Multiplication factor: 1/1 (range usually 1/30 to 1/100)
        if dec > 0:
            self.tau_dec = 1/dec
        else:
            self.tau_dec = 100
        
    def __str__(self):
        return f"ws = {self.ws_day} m/d, Qout = {self.Qout} m^3/s, B = {self.B} m, Qr = {self.Qr} m^3/s"
    
if exp == 'change_Socn' or exp == 'change_Socn_ws':
 #   Socn_range = [5, 10, 15, 20, 25, 30, 35, 40]
    Socn_range = np.linspace(5,100,20)
 #   Socn_range = np.linspace(25,100,15)
elif exp == 'change_Socn_zoom':
    Socn_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
else:
    Socn_range = [30] #ocean salinity
  #  Socn_range = [5] #ocean salinity
    
L = 50e3
q_range = np.nan * np.ones(20)
q_range = np.array(np.linspace(1500, 20500, 20))
testq = 0.00025*3e3*L/q_range

if exp == 'change_B':
 #   B_range = [3e2, 1e3, 3e3, 4e3, 5e3, 7e3, 10e3, 3e4, 3e5]
    B_range = testq*6500/(0.00025*L)
    B_range = np.flip(B_range)
  #  B_range = np.array(np.linspace(900, 13000, 20))
else:
    B_range = [3e3] #width (m)

if exp == 'change_qr':
  #  qr_range = [250,500,1000,2000,3000,4000,5000]
    qr_range = np.array(np.linspace(250, 5000, 20))

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

            #dec = 0 #Decay rate 1/s

            qr = np.zeros(nx-1)
            qr[0] = qr_range[kk]
            Qr = np.cumsum(qr) # river flow [m3/s]
            Qr = np.concatenate(([0], Qr))

            # get the solution at box edges
            Sin, Sout, x, L = rfun.get_Sio_chatwin(Socn, ds, nx)
            #Sin, Sout, x, L = rfun.get_Sio_fjord(Socn, ds, nx)
            Qout = Qr*Sin/(Sin - Sout) #Knudsen 
            Qin = Qr*Sout/(Sin - Sout)

            # run specifications
            ndays = 200 #1000 200
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
            input_tup = (Socn,Qin,Qout,B,Qr,Sin,Sout,L,hs,hd,a0,a1,dt,NT,NS,xm,x,ds,dvs,dvd)

            # intial condition vectors
            csp = np.zeros(NX-1) # csp = "concentration shallow previous"
            cdp = np.zeros(NX-1) # cdp = "concentration deep previous"

            plot_s = False
            if exp == 'salt':
                # Base test case - set to 0 to generate ws_0 cases
                ws = 0 #0.00025
                seq = 0
                dec = 0
                csa, cda, times, csterms, cdterms = rfun.t_calc(csp, cdp, hs, hd, info_tup, dec, ws, seq, riv=1, ocn=0)
            #   plot_s = True
                salt_test = tracer_tests(input_tup,ws,csa,cda,times,csterms,cdterms,seq,dec)
                print('Water flux out at mouth %0.3f [m3/s]' % (-Qout[-1]))
                print('Water flux in at mouth %0.3f [m3/s]' % (Qin[-1]))
                print('Water flux in at head %0.3f [m3/s]' % (Qr[-1]))

            elif exp == 'Tracer_Sink' or exp == 'change_Socn_ws':
                L = 50e03
             #   ws_range = np.array([0, 0.00001, 0.00005, 0.0001, 0.00025, 0.0005, 0.001])
                ws_range = testq*6500/(B*L)
                ws_range = np.flip(ws_range)
               # ws_range = np.array(np.linspace(0, 1.08333333e-03, 20)) #1.08333333e-03 0.0003
                seq = 0
                dec = 0
                
                if exp == 'Tracer_Sink':
                    Tracer_Sink = {}
                elif jj == 0 and exp == 'change_Socn_ws':
                    change_Socn_ws  = {}
                
                for i in range(len(ws_range)):
                    ws = ws_range[i]
                    riv = 1
                    csa, cda, times, csterms, cdterms = rfun.t_calc(csp, cdp, hs, hd, info_tup, dec, ws, seq, riv=riv, ocn=0)
                    if exp == 'Tracer_Sink':
                        Tracer_Sink['ws{0}'.format(i)] = tracer_tests(input_tup,ws,csa,cda,times,csterms,cdterms,seq,dec)
                    else:
                        change_Socn_ws['Socn_ws{0}'.format(len(ws_range)*jj+i)] = tracer_tests(input_tup,ws,csa,cda,times,csterms,cdterms,seq,dec)

            elif exp == 'change_B':
                ws = 0.00025
                seq = 0
                dec = 0
                riv = 1
                
                if ii==0:
                    change_B = {}

                csa, cda, times, csterms, cdterms = rfun.t_calc(csp, cdp, hs, hd, info_tup, dec, ws, seq, riv=riv, ocn=0)
                change_B['B{0}'.format(ii)] = tracer_tests(input_tup,ws,csa,cda,times,csterms,cdterms,seq,dec) 

            elif exp == 'change_Socn' or exp == 'change_Socn_zoom':
                ws = 0.00025
                seq = 0
                dec = 0
                riv = 1
                
                if jj == 0 and exp == 'change_Socn':
                    change_Socn = {}
                elif jj == 0 and exp == 'change_Socn_zoom':
                    change_Socn_zoom = {}

                csa, cda, times, csterms, cdterms = rfun.t_calc(csp, cdp, hs, hd, info_tup, dec, ws, seq, riv=riv, ocn=0)

                if exp == 'change_Socn':
                    change_Socn['Qout{0}'.format(jj)] = tracer_tests(input_tup,ws,csa,cda,times,csterms,cdterms,seq,dec) 
                elif exp == 'change_Socn_zoom':
                    change_Socn_zoom['Qout{0}'.format(jj)] = tracer_tests(input_tup,ws,csa,cda,times,csterms,cdterms,seq,dec)
                else:
                    change_Socn_ws['Socn_ws{0}'.format(len(ws_range)*jj+i)] = tracer_tests(input_tup,ws,csa,cda,times,csterms,cdterms,seq,dec)

            #    print('Water flux out at mouth %0.3f [c m3/s]' % (-Qout[-1]))
            #    print('Water flux in at mouth %0.3f [c m3/s]' % (Qin[-1]))
             #   print('Water flux in at head %0.3f [c m3/s]' % (Qr[-1]))

            elif exp == 'change_qr':
                ws = 0.00025
                seq = 0
                dec = 0
                riv = 1
                if kk==0:
                    change_qr = {}

                csa, cda, times, csterms, cdterms = rfun.t_calc(csp, cdp, hs, hd, info_tup, dec, ws, seq, riv=riv, ocn=0)
                
                change_qr['Qr{0}'.format(kk)] = tracer_tests(input_tup,ws,csa,cda,times,csterms,cdterms,seq,dec)
                
            #    print('Water flux out at mouth %0.3f [c m3/s]' % (-Qout[-1]))
           #     print('Water flux in at mouth %0.3f [c m3/s]' % (Qin[-1]))
            #    print('Water flux in at head %0.3f [c m3/s]' % (Qr[-1]))
    
            elif exp == 'change_seq':
                L = 50e03
                ws = 0.00025
                dec = 0
                riv = 1
          #      seq_range = [0,0.01,0.05,0.1,0.25,0.5,0.75,1]
                seq_range = np.array(np.linspace(0.001, 0.1, 20))

                change_seq = {}
                for i in range(len(seq_range)):
                    seq = seq_range[i]
                    riv = 1
                    csa, cda, times, csterms, cdterms = rfun.t_calc(csp, cdp, hs, hd, info_tup, dec, ws, seq, riv=riv, ocn=0)
                    change_seq['seq{0}'.format(i)] = tracer_tests(input_tup,ws,csa,cda,times,csterms,cdterms,seq,dec) 
                    
            elif exp == 'change_dec':
                L = 50e03
                ws = 0.00025
                seq = 0
                riv = 1
                dec_range = np.array(np.linspace(0, 0.2, 20))

                change_dec = {}
                for i in range(len(dec_range)):
                    dec = dec_range[i]
                    riv = 1
                    csa, cda, times, csterms, cdterms = rfun.t_calc(csp, cdp, hs, hd, info_tup, dec, ws, seq, riv=riv, ocn=0)
                    change_dec['dec{0}'.format(i)] = tracer_tests(input_tup,ws,csa,cda,times,csterms,cdterms,seq,dec)
                    
            else:
                print('exp = %s not supported' % (exp))
        #        sys.exit()
            X = x/1e3
            Xm = xm/1e3

# PLOTTING
plt.close('all')
'''
fs = 16
lw = 3
plt.rc('font', size=fs)
#fig = plt.figure(figsize=(12,10))
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
    ax.text(.1, .6, 'Dotted = Target Solution', #transform=ax.transAxes)
    ax.text(.1, .5, 'Solid = Numerical Solution', #transform=ax.transAxes)
ax.grid(True)
ax.set_xlim(0,X[-1])
ax.set_ylabel('Tracer')   
ax.text(.2, .8, r'$Deep$', color=cin, transform=ax.transAxes, #size=1.5*fs,
    bbox=dict(facecolor='w', edgecolor='None', alpha=0.5))
ax.text(.4, .8, r'$Shallow$', color=cout, transform=ax.transAxes, size=1.5*fs,
    bbox=dict(facecolor='w', edgecolor='None', alpha=0.5))
ax.set_ylim(0, 1.1*np.max([cda[-1,:].max(),csa[-1,:].max()]))
ax.set_xlabel('X (km)')
ax.set_title(exp.replace('_',' ')+' ws=%f m/s' %ws)

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
ax.set_xlim(0,X[-1])
ax.set_ylim(0, 1.1)
'''
