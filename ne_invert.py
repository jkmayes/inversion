import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import numpy.matlib
import scipy
import scipy.interpolate
import scipy.io as sio
from scipy.optimize import minimize
import cdflib
import os
import matplotlib.colors as colors
import DateTimeTools as TT
import PyFileIO as pf  
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
import matplotlib.ticker as plticker
import matplotlib.dates as mdates
import fnmatch as fnm
import DateTimeTools as dtt 
import os as os
import pytz
import pdb
import copy
#pad = Arase.LEPe.ReadPAD(20180101,'eFlux')
#E=[ 0.06696621,  0.09084448,  0.11472277,  0.14457063,  0.18038803,
#        0.22217506,  0.2759012 ,  0.34156644,  0.42514044,  0.52065355,
#        0.6400449 ,  0.7892842 ,  0.96240187,  1.1773064 ,  1.4399675 ,
#        1.7623242 ,  2.1503465 ,  2.6219423 ,  3.200991  ,  3.9054003 ,
#        4.7650185 ,  5.8096933 ,  7.081211  ,  8.633301  , 10.525654  ,
#       12.829912  , 15.635605  , 19.056175  ]*10
E=np.logspace(2,6.5,25)

#massarr=[1.272E-03,3.932E-04,8.787E-05,1.959E-05,4.589E-06,1.312E-06,4.169E-07,1.220E-07,3.028E-08,3.628E-09,3.242E-10,6.532E-11,1.717E-11,6.117E-12,2.914E-12,1.609E-12,9.690E-13,6.167E-13,4.077E-13,2.772E-13,1.925E-13,1.359E-13,9.726E-14,7.045E-14,5.156E-14,3.810E-14,2.840E-14,2.135E-14,1.618E-14,1.235E-14,9.510E-15,7.361E-15,5.734E-15,4.494E-15,3.543E-15,2.807E-15,2.236E-15,1.788E-15,1.437E-15,1.159E-15,9.376E-16,7.591E-16,6.184E-16,5.051E-16,4.135E-16,3.393E-16,2.789E-16,2.297E-16,1.896E-16,1.567E-16,1.297E-16,1.075E-16,8.929E-17,7.425E-17,6.182E-17,5.154E-17,4.304E-17,3.599E-17,3.014E-17,2.527E-17,2.123E-17,1.786E-17,1.506E-17,1.272E-17,1.076E-17,9.125E-18,7.757E-18,6.610E-18,5.648E-18,4.840E-18,4.160E-18,3.588E-18,3.105E-18,2.698E-18,2.353E-18,2.061E-18,1.813E-18,1.602E-18,1.422E-18,1.269E-18,1.137E-18,1.024E-18,9.273E-19,8.434E-19,7.708E-19,7.077E-19,6.526E-19,6.045E-19,5.622E-19,5.249E-19,4.919E-19,4.625E-19,4.363E-19,4.128E-19,3.916E-19,3.725E-19,3.551E-19,3.392E-19,3.247E-19,3.114E-19,2.991E-19]
massarr=[1.272,3.932E-01,8.787E-02,1.959E-02,4.589E-03,1.312E-03,4.169E-04,1.220E-04,3.028E-05,3.628E-06,3.242E-7,6.532E-8,1.717E-8,6.117E-9,2.914E-9,1.609E-9,9.690E-10,6.167E-10,4.077E-10,2.772E-10,1.925E-10,1.359E-10,9.726E-11,7.045E-11,5.156E-11,3.810E-11,2.840E-11,2.135E-11,1.618E-11,1.235E-11,9.510E-12,7.361E-12,5.734E-12,4.494E-12,3.543E-12,2.807E-12,2.236E-12,1.788E-12,1.437E-12,1.159E-12,9.376E-13,7.591E-13,6.184E-13,5.051E-13,4.135E-13,3.393E-13,2.789E-13,2.297E-13,1.896E-13,1.567E-13,1.297E-13,1.075E-13,8.929E-14,7.425E-14,6.182E-14,5.154E-14,4.304E-14,3.599E-14,3.014E-14,2.527E-14,2.123E-14,1.786E-14,1.506E-14,1.272E-14,1.076E-14,9.125E-15,7.757E-15,6.610E-15,5.648E-15,4.840E-15,4.160E-15,3.588E-15,3.105E-15,2.698E-15,2.353E-15,2.061E-15,1.813E-15,1.602E-15,1.422E-15,1.269E-15,1.137E-15,1.024E-15,9.273E-16,8.434E-16,7.708E-16,7.077E-16,6.526E-16,6.045E-16,5.622E-16,5.249E-16,4.919E-16,4.625E-16,4.363E-16,4.128E-16,3.916E-16,3.725E-16,3.551E-16,3.392E-16,3.247E-16,3.114E-16,2.991E-16]
#units above!!!!!!!

#maybe better ways than hard codeing this in but not sure if i want to have to import Arase module

def FileSearch(dirname,fname):
    if os.path.isdir(dirname) == False:
        return np.array([])
		
    files=np.array(os.listdir(dirname))
    files.sort()
    matches=np.zeros(np.size(files),dtype='bool')
    for i in range(0,np.size(files)):
        if fnm.fnmatch(files[i],fname):
            matches[i]=True
	
    good=np.where(matches == True)[0]
    if np.size(good) == 0:
        return np.array([])
    else:
        return np.array(files[good])

def get_data(date):
    #fpath ='Documents/eiscat/'
    #fname =  FileSearch(fpath,'NCAR_'+str(date)[0:4]+'-'+str(date)[4:6]+'-'+str(date)[6:8]+'*')
    #print('NCAR_'+str(date)[0:4]+'-'+str(date)[4:6]+'-'+str(date)[6:8])
    #print(fname)
    fpath ='Documents/eiscat/'
    fname =  FileSearch(fpath,'MAD6400_'+str(date)[0:4]+'-'+str(date)[4:6]+'-'+str(date)[6:8]+'*')
    print('MAD6400_'+str(date)[0:4]+'-'+str(date)[4:6]+'-'+str(date)[6:8])
    print(fname)
    data = pf.ReadASCIIData(fpath + fname[0],Missing=['missing','assumed'],dtype=[('YEAR', '<i4'), ('MONTH', '<i4'), ('DAY', '<i4'), ('HOUR', '<i4'), ('MIN', '<i4'), ('SEC', '<i4'), ('RECNO', '<i4'), ('KINDAT', '<i4'), ('KINST', '<i4'), ('UT1_UNIX', 'float64'), ('UT2_UNIX', 'float64'), ('HSA', '<f4'), ('AZM', '<f4'), ('ELM', '<f4'), ('POWER', '<f4'), ('SYSTMP', '<f4'), ('RANGE', '<f4'), ('GDALT', '<f4'), ('NE', '<f4'), ('TI', '<f4'), ('TR', '<f4'), ('CO', '<f4'), ('VO', '<f4'), ('PM', '<f4'), ('PO+', '<f4'), ('DNE', '<f4'), ('DTI', '<f4'), ('DTR', '<f4'), ('DCO', '<f4'), ('DVO', '<f4'), ('DPM', '<f4'), ('DPO+', '<f4'), ('DWN', '<f4'), ('DDC', '<f4'), ('GFIT', '<f4'), ('CHISQ', '<f4')])
    #remove bad recno
    arr = data['GDALT']>400
    use = np.where(arr == 0)[0]
    data = data[use]

    return data

class eiscat_inversion(object):
    def __init__(self,date,**kwargs):
        #line getting data gose here
        #keys = list(filedict.keys())
		#for k in keys:
		#setattr(self,k,filedict[k])
        data=get_data(date)
        print(data)
        self.date=date
        self.HOUR=data['HOUR']
        self.HOUR = np.array([np.array(self.HOUR[np.where(data.RECNO == u)[0]]) for u in np.unique(data.RECNO)])
        self.MIN=data['MIN']
        self.MIN = np.array([np.array(self.MIN[np.where(data.RECNO == u)[0]]) for u in np.unique(data.RECNO)])
        self.lat=69.6
        self.long=19.2
        self.topatmos=0
        self.E=E #pad.Emid[3][0:-4] #breake this and de into diffrent instuments/ alt overlap and find diffrence   
        self.dE=np.gradient(self.E); # bin widths 
        self.RECNO=data['RECNO'] 
        self.alt=data['GDALT']*1000#units here!!!!!!!!!
        self.alt_prof=np.array([np.array(self.alt[np.where(data.RECNO == u)[0]]) for u in np.unique(data.RECNO)])
        self.ti=data['TI']
        self.tr=data['TR']
        self.Te=self.tr*self.ti
        self.Te_prof=np.array([np.array(self.Te[np.where(data.RECNO == u)[0]]) for u in np.unique(data.RECNO)])
        self.particle='electron'
        self.recomb='guess i can put anything here'
        #self.ne=np.power(10,data['NEL']) #this is log needs to not be!!!
        self.ne=data['NE']
        #self.e_ne=np.power(10,data['DNEL'])
        self.e_ne=data['DNE']
        print(data['DNE'])
        print(self.e_ne)
        self.ne_prof = np.array([np.array(self.ne[np.where(data.RECNO == u)[0]]) for u in np.unique(data.RECNO)])
        self.e_ne_prof = np.array([np.array(self.e_ne[np.where(data.RECNO == u)[0]]) for u in np.unique(data.RECNO)])
        self.timedep=0
        self.plot=0
        self.invmethod='semeter'
        self.erc_method='vickery'
        self.e_range_method='sk05'
        self.massarr=massarr
        self.ut_unix=(data['UT1_UNIX'])-3600
        self.ut_unix = np.array([np.array(self.ut_unix[np.where(data.RECNO == u)[0]]) for u in np.unique(data.RECNO)])
        self.ut_unix=self.ut_unix[:,0]#-3600
        self.bad_fit_on=False
        self.bad_fit=[]
        self.ucr=15#the chunk ranges
        self.lcr=0#the chunk ranges
        #chapman model param
        self.chapman_max_density=[2.1E11]
        self.chapman_max_den_height=[240000]#height of maximum density
        self.scale_height=50000
        #self.hight_list=np.arange(200000,400000,1000)
        self.hhh=(self.alt_prof[1]-self.chapman_max_den_height)/self.scale_height

    def top_of_atmosphere_mass(self):
        if self.topatmos==False:
            intmass0=0
        else:
            print('dont do that right now')
            #find a python equivalnte for this 
            #intmass0=integrate_atmos(z(end),par.topatmos,par.t,par.lat,par.long,par.f107a,par.f107p,par.ap);

    def get_msis_data(self):
        #print('dont do that right now')
        #He,O,N2,O2,Ar,Mass,H,N,Tex,T=msis(UTC,[z_min z_max]/1e3,self.lat,self.long,f107a,f107p,ap)

        #get the data and add it to the object
        altarr=np.arange(0,1000001,10000)
        f = scipy.interpolate.interp1d(altarr,self.massarr,bounds_error=False,fill_value=0)
        #pdb.set_trace()
        self.mass=f(self.alt_prof[0])
        #self.mass=[2.901E-08, 5.587E-09, 7.676E-10, 1.671E-10, 5.553E-11, 1.796E-11, 7.610E-12, 3.973E-12, 2.319E-12,1.454E-12,9.574E-13, 6.527E-13, 4.567E-13, 3.261E-13, 2.365E-13, 1.738E-13, 1.291E-13, 9.683E-14, 7.324E-14, 
#5.582E-14, 4.284E-14, 3.310E-14, 2.574E-14, 2.013E-14, 1.584E-14,1.253E-14,9.962E-15,7.963E-15]

#im gonna foucus on geting the sk05 method and electrons working to start with -Joe
    def electron_ionisation(self,s0=0):
        z=self.alt_prof[0]
        K=self.E
        rho=self.mass
        method=self.e_range_method
        # initialise output matrix
        Q=np.matlib.repmat(np.nan,len(z),len(K))
        #Q=np.full([len(z),len(k)], np.nan)
        # mass distance travelled to altitude z
        if any(np.diff(z)<=0):
             print('z must be strictly increasing')

        z1=np.flipud(z)
        rho=np.flipud(rho)
        s0=0
        s=-scipy.integrate.cumtrapz(z1,rho)#scipy.integrate.
        #s=s+s0
        s = np.insert(s, 0, 0, axis=0)


        #method choice
        if method == 'sk05':
            # electron range
            R=self.electron_range(K)

            # loop for each energy
            for j in range(0,len(K)):
                # lambda function
                l=self.semeter_lambda((s/R[j]))
                q=l * rho *K [j] / (35.5 * R[j])
                q=np.flipud(q)
                Q[:,j]=q[:]
            #pdb.set_trace()
            return Q

        ####add method si93 later#####


    def electron_range(self,K):
        # ELECTRON_RANGE  Range of an electron in air
        # where K is the initial energy (eV) and R is in mass distance (kg m^-2).
        # If METHOD is specified, it can be one of:
        #
        # 'sk05':
        #   Formula from Semeter and Kamalabadi (2005), Radio Sci.,
        #   40, RS2006, doi:10.1029/2004RS003042. Originally from Barrett and
        #   Hays, 1976.
        #
        # 'si93':
        #   From Sergienko & Ivanov (1993), Ann. Geophys., 11, 717.
        #
        # 'gj84':
        #		From Goldberg et al (1984), JGR, 89, 5581.
        #
        # 'grmix':
        #		A mixture of Goldberg et al and Rees
        #
        # If not specified, the default is 'sk05'.
        #
        # For 'si93', an additional parameter must be specified to select the
        # pitch angle model: 'idh' for isotropic over the downward
        # hemisphere or 'mono' for monodirectional.
        
        
        if self.e_range_method == 'sk05':
            K=np.array(K)
            R=4.30e-6+5.36e-5*(K/1000)**1.67
            return R
        ###sort other methods later###

    def erc(self):
        # 'method': method of calculation. Options are
        #   'vickrey' - Vickrey et al, quoted by Semeter & Kamalabadi, RS,
        #   2005
        #
        #   'gledhill_na' - "night aurora" profile from Gledhill, RS, v.21,
        #   399, 1986.
        #
        #   'gledhill_daymix' - (from Gledhill, RS, v21, 399, 1986)
        #     This corresponds to the "aurora" profile (Fig. 1) above
        #     about 90 km and the "PCA, SPA, flare daytime" (Fig. 3)
        #     profile below about 90 km. The intention is to simulate a
        #     realistic daytime profile over a wide range of altitudes.
        #
        #   'rodger_night' - from Rodger et al., JGR, VOL. 112, A11307,
        #     doi:10.1029/2007JA012383, 2007
        #
        #   'rodger_day' - from Rodger et al., JGR, VOL. 112, A11307,
        #     doi:10.1029/2007JA012383, 2007
        #
        # 'alt': vector/matrix of altitudes in metres
        #
        # 'Te': vector/matrix of electron temperatures in Kelvin (necessary for Rodger)
        #
        # Outputs:
        #
        #  alpha - effective recombination coefficient in m^3 s^-1.
        z=self.alt_prof[0]
        T=self.Te_prof
        #alpha = np.full(np.size(T),np.nan)

        alpha = np.matlib.repmat(np.nan,np.shape(T)[0],np.shape(T)[1])
        if self.erc_method =='vickery':
            # Recombination coefficient (from Vickrey et al, quoted by Semeter & Kamalabadi, 2005)
            alpha=2.5e-12* np.exp(-(z/1e3)/51.2)
            return alpha
    def intergrate_atmos(z_min,z_max,UTC,lat,long,f107a,f107p,ap):
        print('ohno')
    #need the msis data for this bit
       # He,O,N2,O2,Ar,Mass,H,N,Tex,T=msis(UTC,[z_min z_max]/1e3,self.lat,self.long,f107a,f107p,ap)
        ##why get all the things when it seems only mass is needed
        #Mass=Mass*1e3 # convert from g cm^-3 to kg m^-3
        #a=log(Mass[-1]/Mass[1]) / (z_max-z_min)
        #y0=Mass[1] / exp(a * z_min)
        #I_est=(y0 / a) * (exp(a * z_max) - exp( a * z_min))
        #fprintf('Exponential estimate of integral = %.2g kg/m^2\n',I_est)
        #fprintf('Average scale height = %.2f km\n',-1e-3/a)

    def semeter_lambda(self,x):

        #print (dummy)
        ltab=[0.000,0.006,0.020,0.027,0.042,0.058,0.069,0.091,0.123,0.145,0.181,0.214,0.248,
        0.313,0.360,0.431,0.499,0.604,0.728,0.934,1.237,1.686,1.995,2.063,2.024,1.946,1.846,
        1.761,1.681,1.596,1.502,1.421,1.346,1.260,1.190,1.101,1.043,0.972,0.888,0.834,0.759,
        0.689,0.636,0.567,0.504,0.450,0.388,0.334,0.282,0.231,0.187,0.149,0.113,0.081,0.058,
        0.037,0.023,0.010,0.002,0.000]
        xtab=np.arange(-0.525,0.975,0.025)#(-0.525:0.025:0.950)
        #interpolate with chosen method, but return zero outside of range
        #l=interp1d(xtab,ltab,x,method,0);
        f = scipy.interpolate.interp1d(xtab,ltab,bounds_error=False,fill_value=0)
        l=f(x)
        #pdb.set_trace()
        return l

    def mem_iter_joe(self,phi,A,q,w,beta):
        #print(dummy,phi,A,q,w,beta)
        Nphi=len(phi)
        Nq=np.size(A,0) 
        #equation A3
        #t=np.full([Nq,1],np.nan)
        t=np.matlib.repmat(np.nan,Nq,1)
        for i in range(0,Nq):
            a=A[i,:]
            a[np.where(a==0)[0]]=np.nan
            var=1/a*phi.T
            t[i]=np.min(var)#(1 / (a * phi))
        t[np.isnan(t)]=0
        #pdb.set_trace()
        
        #equation A2
        #c=np.full([A,1],np.nan)
        c=np.matlib.repmat(np.nan,Nq,1)
        for i in range(0,Nq):
            m1=1-A[i]
            m2=phi/q[i]
            mm=np.matmul(m1,m2)
            c[i]=beta * mm * t[i]
            #c[i]=beta * (1-A[i] * phi/q[i]) * t[i]
        #pdb.set_trace()
        #equation A1
        phi_new=np.full(np.shape(phi),np.nan)
        #phi_new=np.matlib.repmat(np.nan,np.shape(phi))
        for j in range(0,Nphi):
            m1=(1-phi[j])
            m2=np.sum(w * c * A[:,j])
            #mm=np.matmul(m1,m2)
            #pdb.set_trace()
            phi_new[j]= phi[j] / (1-phi[j] * np.nansum(w * c * A[:,j]))
            #pdb.set_trace()#phi_new(j)=phi(j)/(1-phi(j)*sum(w.*c.*A(:,j)))
        #pdb.set_trace()
        return phi_new
#matts version
    def mem_iter(self,phi,A,q,w,beta):
        Nphi = phi.shape[0]
        Nq = A.shape[0]

        t = np.zeros((Nq,1),dtype='float64') + np.nan
        for i in range(0,Nq):
                a = np.copy(A[i,:])
                a[a==0] = np.nan
                t[i] = np.nanmin(1/(a*phi.T))
        t[np.isnan(t)]=0

        #equation A2
        c = np.zeros((Nq,1),dtype='float64') + np.nan
        for i in range(0,Nq):
                c[i]=beta*(1-np.dot(A[i,:],phi/q[i]))*t[i]

        #equation A1
        phi_new = np.zeros(phi.shape,dtype='float64') + np.nan
        for j in range(0,Nphi):
                phi_new[j]= phi[j] / (1-phi[j] * np.nansum(w[:,0] * c[:,0] * A[:,j]))
        #pdb.set_trace()
        #print(c)
        return phi_new



    def FitFunction(self,logf,Q0,q_obs,e_q_obs):
	#This function will be minimized
        f=10**logf
        q_inv = np.dot(Q0,f)
        chi2 = np.nansum(((q_obs-q_inv)/e_q_obs)**2)
        return chi2





    #do the actual inversion
    def inversion(self):
        ff=np.array([])
        self.af=np.zeros([self.ne_prof.shape[0],len(self.E)])
        F=np.zeros([25,self.ne_prof.shape[0]])#this is the final output
        self.get_msis_data()
        if self.particle== 'electron':
            Q00=self.electron_ionisation(self)#,z,E,Mass,[],intmass0)
            self.Q00=Q00
        # scale according to the bin widths
        #Q00=Q00*np.matlib.repmat(self.dE,np.size(Q00,1),1)#figure this out 

        #effective recombination coefficient
        if self.recomb=='NO+':
            # From Walls & Dunn, quoted by Semeter & Kamalabadi
            alpha=4.2e-13*(300/self.Tn)**0.85
        elif self.recomb=='02+':
            # From Walls & Dunn, quoted by Semeter & Kamalabadi 
            alpha=1.95e-13*(300/self.Tn)**0.7 
        else:
            alpha=self.erc()

        # Convert densities (and errors) to production rates (and errors)
        q=alpha * self.ne_prof ** 2
        e_q=2 * alpha * self.ne_prof * self.e_ne_prof
        #time dependent stuff??

        if self.timedep==True:
            [dn,vdn]=deriv(t,self.ne,self.e_ne ** 2)
            q=q + dn
            e_q=sqrt(e_q ** 2 + vdn)

        #if self.plot==True:
        #    ax=figure

        for k in range(0,self.ne_prof.shape[0]):
            q_obs=q[k]#check what this bit means
            e_q_obs=e_q[k]
            nomask=q_obs
            #if par.plot
            #myerrorbar(1:length(q_obs),q_obs,e_q_obs);
            #title(sprintf('#d',k));
            #if this makes the error bar i  might put it later 

            # Delete altitudes with no or unusable data
            mask= ~np.isnan(q_obs) & (q_obs>0)
            q_obs=q_obs[mask]
            e_q_obs=e_q_obs[mask]
            Q0=Q00[np.where(mask)[0],:]
            #check with matt about masking 
            self.Q0=Q0
            self.q_obs=q_obs
            self.e_q_obs=e_q_obs
            # Form matrices for the inversion process
            if self.invmethod=='semeter':
                # iterative MEM solution        
                Q0[np.isnan(Q0)] = 0.0
                #k=0
                beta=2
                w = np.ones((len(q_obs),1),dtype='float64')/len(q_obs)
                f = 1e11*np.ones((Q0.shape[1],1),dtype='float64')
                max_iter=90
                chi2_thresh=0.5e-4
                q_inv=np.matmul(Q0,f)
                chi2 = np.nansum(((q_obs-q_inv)/e_q_obs)**2)
                fit = False
                #pdb.set_trace()
                self.Q0=Q0
                self.q_obs=q_obs
                self.e_q_obs=e_q_obs
                for n in range (0,max_iter):
                    #print(f)
                    #plt.plot(f)
                    #pdb.set_trace() 
                    f=self.mem_iter(f,Q0,q_obs,w,beta)
                    q_inv = np.matmul(Q0,f)
                    old_chi2 = copy.deepcopy(chi2)
                    chi2 = np.nansum(((q_obs-q_inv)/e_q_obs)**2)
                    a =np.abs(old_chi2-chi2)
                    stop_val = chi2_thresh * old_chi2
                    #pdb.set_trace()
                    print('\rn: {:d}, χ2: {:8.3f}'.format(n,chi2),end='') 
                    if a<stop_val:
                        fit = True
                        break
                print()
                if fit:
                    print('k: {:d}, n: {:d} χ2: {:8.3f}\n'.format(k,n,chi2))
                else:
                    print('No fit done\n')
                    self.bad_fit.append(k)
                
                q_inv=np.matmul(Q0,f) #why is this repeated here?
                chi2=sum(((q_obs - q_inv) / e_q_obs) ** 2)
                #F[k]=f#np.append(F,f)#[:,k]=f
                #pdb.set_trace()
                self.af[k]=f[:,0]
                if self.plot==True:
                    plt.figure()
                    plt.plot(q_obs,self.alt_prof[0,mask],label='$q_{obs}$')
                    plt.plot(q_inv,self.alt_prof[0,mask],label='scipy $q_{inv}$')
                    plt.gca().set_xlim(1,1e10)
                    plt.legend()
                    plt.savefig('time='+str(self.HOUR[k,0])+'_'+str(self.MIN[k,0])+'.png',)
                    plt.close()
                #ff=np.append(ff,f)
                #CHI2[k]=chi2;
            elif self.invmethod=='L-BFGS-B':
                Q0[np.isnan(Q0)] = 0.0
                f0 = 10*np.ones((Q0.shape[1],1),dtype='float64')
                #create callback function
                MaxIter = 1000
                chi2 = []
	
                def callback(*args):
                    fk = args[0]
                    q_inv = np.dot(Q0,fk)
                    c2 = np.nansum(((q_obs-q_inv)/e_q_obs)**2)
                    chi2.append(c2)
                res = minimize(self.FitFunction,f0,args=(Q0,q_obs,e_q_obs),method='L-BFGS-B',options={'ftol': 2.220446049250313e-9},callback=callback)
                #get the result
                print('Success: ',res.success)
                print('Iterations: ',res.nit)
                f = 10**res.x
                self.af[k]=f
                #self.ff=np.append(ff,f)
                #print(f)
	        #calculate the inverted spectrum
                q_inv = np.dot(Q0,f)
                if self.plot==True:
                    plt.figure()
                    plt.plot(q_obs,self.alt_prof[0,mask],label='$q_{obs}$')
                    plt.plot(q_inv,self.alt_prof[0,mask],label='scipy $q_{inv}$')
                    plt.gca().set_xlim(1,1e10)
                    plt.legend()
                    plt.savefig('time='+str(self.HOUR[k,0])+'_'+str(self.MIN[k,0])+'.png',)
                    plt.close()
                #pdb.set_trace()
        self.ff0=ff
        #self.ff=np.reshape(ff,[len(E),self.ne_prof.shape[0]])
        print(np.shape(Q00),np.shape(self.af))
        self.inv_ne = np.sqrt(np.dot(Q00,self.af.T).T/alpha).T
        #self.inv_ne=(np.matmul(self.ff,Q00))
        #for i in range(0,np.shape(self.ff)[1]):
        #    self.inv_ne[:,i]=(self.inv_ne[:,i]/alpha)**0.5 
    def spec_plot(self,index,save=False):
        plt.figure()
        plt.plot(self.E*0.001,self.af[index],label=self.invmethod)
        pdb.set_trace()
        plt.xlabel('energy(Kev)', fontsize=18)
        plt.ylabel('flux', fontsize=18)
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        if save==True:
            plt.savefig('flux_at_time='+str(self.HOUR[index,0])+'_'+str(self.MIN[index,0])+'.png',)
            plt.close('all')
    def ne_plot(self,index,save=False):
        plt.figure()
        plt.plot(self.ne_prof[index],self.alt_prof[index]/1000,label='raw density')
        plt.plot(self.inv_ne[:,index],self.alt_prof[index]/1000,label='inverted density'+str(self.invmethod))
        #plt.gca().set_xlim(1,1e10)
        plt.xlabel('density', fontsize=18)
        plt.ylabel('altitude \n (km)', fontsize=18)
        plt.legend()
        if save==True:
            plt.savefig('density_at_time='+str(self.HOUR[index,0])+'_'+str(self.MIN[index,0])+'.png',)
            plt.close('all')
    def save_all_plots(self,n,flux=1,ne=1):
        for i in range(0,n-1):
            if flux==True:
                self.flux_plot(i,save=True)
            if ne==True:
                self.ne_plot(i,save=True)
    def diff_calc(self,n):
        for i in range(0,n-1):
            self.diff=np.zeros((n,len(self.alt_prof[0])))
            self.diff[i,:]=np.abs(self.ne_prof[i]-self.inv_ne[:,i])
    def density_plot(self,fig=None,maps=[1,1,0,0]):
        if fig is None:
            fig = plt
            fig.figure()
        if hasattr(fig,'Axes'):	
            ax = fig.subplot2grid((maps[1],maps[0]),(maps[3],maps[2]))
        else:
            ax = fig
        self.ax = ax
        #diff=self.diff[:-1,:-1]
        sm=ax.pcolormesh(self.datetime,self.alt_prof[0]/1000,np.log10(self.ne_prof.T),cmap='gist_ncar',vmin=10,vmax=12)
        plt.xlabel('UT', fontsize=18)
        plt.ylabel('altitude \n (km)', fontsize=18)
        #colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.05)	
        cbar = fig.colorbar(sm,cax=cax) 
        cbar.set_label('log density')
    def model_plot(self,fig=None,maps=[1,1,0,0]):
        if fig is None:
            fig = plt
            fig.figure()
        if hasattr(fig,'Axes'):	
            ax = fig.subplot2grid((maps[1],maps[0]),(maps[3],maps[2]))
        else:
            ax = fig
        self.ax = ax
        #diff=self.diff[:-1,:-1]
        sm=ax.pcolormesh(self.datetime,self.alt_prof[0]/1000,np.log10(self.model_density.T),cmap='gist_ncar',vmin=10,vmax=12)
        plt.xlabel('UT', fontsize=18)
        plt.ylabel('altitude \n (km)', fontsize=18)
        #colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.05)	
        cbar = fig.colorbar(sm,cax=cax) 
        cbar.set_label('log density')
    def inv_plot(self,fig=None,maps=[1,1,0,0]):
        if fig is None:
            fig = plt
            fig.figure()
        if hasattr(fig,'Axes'):	
            ax = fig.subplot2grid((maps[1],maps[0]),(maps[3],maps[2]))
        else:
            ax = fig
        self.ax = ax
        #diff=self.diff[:-1,:-1]
        if self.bad_fit_on==True:
            inv_ne_val=self.inv_ne
            inv_ne_val[:,self.bad_fit]=np.nan
        else:
            inv_ne_val=self.inv_ne
        sm=ax.pcolormesh(self.datetime,self.alt_prof[0]/1000,np.log10(inv_ne_val),cmap='gist_ncar',vmin=10,vmax=12)
        plt.xlabel('UT', fontsize=18)
        plt.ylabel('altitude \n (km)', fontsize=18)
        #colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.05)	
        cbar = fig.colorbar(sm,cax=cax) 
        cbar.set_label('log inverted density')
    def diff_plot(self,var='inv_ne',fig=None,maps=[1,1,0,0]):
        if var=='inv_ne':
            vari=self.inv_ne
        else:
            vari=self.model_density
        if fig is None:
            fig = plt
            fig.figure()
        if hasattr(fig,'Axes'):	
            ax = fig.subplot2grid((maps[1],maps[0]),(maps[3],maps[2]))
        else:
            ax = fig
        self.ax = ax
        #diff=self.diff[:-1,:-1]
        sm=ax.pcolormesh(self.datetime,self.alt_prof[0]/1000,(np.abs(vari.T-self.ne_prof.T))/self.ne_prof.T,cmap='gist_ncar')
        plt.xlabel('UT', fontsize=18)
        plt.ylabel('altitude \n (km)', fontsize=18)
        #colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.05)	
        cbar = fig.colorbar(sm,cax=cax) 
        cbar.set_label('% diffrence')
    def flux_plot(self,fig=None,maps=[1,1,0,0]):
        #for individual plots use plt.plot(np.log10(obj.E),np.log10(obj.af[0]))
        if fig is None:
            fig = plt
            fig.figure()
        if hasattr(fig,'Axes'):	
            ax = fig.subplot2grid((maps[1],maps[0]),(maps[3],maps[2]))
        else:
            ax = fig
        self.ax = ax
        #diff=self.diff[:-1,:-1]
        sm=ax.pcolormesh(self.datetime,np.log10(self.E),np.log10(self.af.T),vmin=6,vmax=14,cmap='gnuplot')
        plt.xlabel('UT', fontsize=18)
        plt.ylabel('log(energy) \n eV', fontsize=18)
        #colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.05)	
        cbar = fig.colorbar(sm,cax=cax) 
        cbar.set_label('log(flux)')
    def density_panel(self):
        fig = plt  
        fig.figure()
        self.density_plot(fig=plt,maps=[1,3,0,0])
        self.inv_plot(fig=plt,maps=[1,3,0,1])
        self.diff_plot(fig=plt,maps=[1,3,0,2])
    def flux_panel(self):
        fig = plt  
        fig.figure()
        self.density_plot(fig=plt,maps=[1,3,0,0])
        self.inv_plot(fig=plt,maps=[1,3,0,1])
        self.flux_plot(fig=plt,maps=[1,3,0,2])
    def model_panel(self):
        fig = plt  
        fig.figure()
        self.density_plot(fig=plt,maps=[1,3,0,0])
        self.model_plot(fig=plt,maps=[1,3,0,1])
        self.diff_plot(fig=plt,maps=[1,3,0,2],var=self.model_density)
    def unix_to_datetime(self):
        """
        Convert unix time array to datetime array
        """
        datetime_array = []
        for unix_time in self.ut_unix:
            datetime_array.append(datetime.fromtimestamp(unix_time))
        self.datetime= datetime_array 
    def chunk_plot(self,chap=False):
        for i in range(self.lcr,self.ucr):
            plt.plot(self.ne_prof[i],self.alt_prof[i]/1000)
            #plt.xlabel('density (m\u00b3)', fontsize=18)
            plt.ylabel('altitude \n (km)', fontsize=18)
            plt.xlabel(r'density $\mathrm{(m}^{\ -3  })$',fontsize=18)
            if chap==True:
                plt.plot(self.model_density[i],self.alt_prof[i]/1000)
    def chap_model(self):
        self.model_density=(self.chapman_max_density) * np.exp(0.5*  (1  -self.hhh  -np.exp(-self.hhh)))
        plt.plot(self.model_density,self.alt_prof[1]/1000)
        plt.xlabel('density (m^-3)', fontsize=18)
        plt.ylabel('altitude \n (km)', fontsize=18)
    def background_subtraction(self):
        self.ne_prof=self.ne_prof-self.model_density
        self.ne_prof[np.where(self.ne_prof<1)]= 1e10
    def multi_chap(self):
        #good_alts=self.ne_prof[0,np.where(200000<self.alt_prof[0])]
        self.model_density=np.zeros(np.shape(self.ne_prof))
        for i in range(0,len(self.ne_prof)):
            good_dens=self.ne_prof[i,20:27]
            good_alts=self.alt_prof[i,20:27]
            Max_Den=np.amax(good_dens)
            Max_Alts=good_alts[np.argmax(good_alts)-1]
            multi_H=(self.alt_prof[i]-Max_Alts)/self.scale_height
            self.hhh=(self.alt_prof[1]-self.chapman_max_den_height)/self.scale_height
            chap=(Max_Den) * np.exp(0.5*  (1  -multi_H  -np.exp(-multi_H)))  
            #print(multi_H,Max_Den,Max_Alts)spec = Arase.Electrons.ReadOmni(20170621) 
            self.model_density[i]=chap
    def smooth_af(self):
        print("not working yet")
    def line_calc(self):
        import Arase
        pad1=Arase.LEPe.ReadPAD(self.date,'eFlux')
        pad2=Arase.MEPe.ReadPAD(self.date,'eFlux')
        self.lep_line=np.nansum(pad1.Flux[:,:,0],axis=1)
        self.mep_line=np.nansum(pad2.Flux[:,:,0],axis=1)
        loss1=(pad1.Flux[:,:,0])#fine tune this later
        bin_width1=(pad1.Emin[0]-pad1.Emax[0])
        loss2=(pad2.Flux[:,:,0])#fine tune this later
        bin_width2=(pad2.Emin[0]-pad2.Emax[0])
        #get energy
        self.energy_specL=loss1*bin_width1*pad1.Emid
        self.energy_specM=loss2*bin_width2*pad2.Emid
        self.LEP_energy=np.nansum(self.energy_specL,axis=1)
        self.MEP_energy=np.nansum(self.energy_specM,axis=1)
        self.Mline=np.nansum(self.energy_specM,axis=1)
        self.Lline=np.nansum(self.energy_specL,axis=1)


    def alt_set(self,laltbin,ualtbin):
        self.alt_prof=self.alt_prof[:,laltbin:ualtbin]
        self.Te_prof=self.Te_prof[:,laltbin:ualtbin]
        self.e_ne_prof=self.e_ne_prof[:,laltbin:ualtbin]
        self.ne_prof=self.ne_prof[:,laltbin:ualtbin]
        
#Arase stuff
#spec.PlotSpectrum(20170621,23.50,xparam='E',yparam='Flux',Split=True)
#spec = Arase.Electrons.ReadOmni(20170621) 
#plt.plot(obj.E*10**-3,obj.af[-5]*10**-2.5)   
