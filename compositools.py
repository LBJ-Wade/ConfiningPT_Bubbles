""" Useful functions and definitions for the relic abundance calculation for composite dark sectors
...
"""
import sys
import csv
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import scipy.special as special
from scipy.integrate import odeint
from scipy.interpolate import RegularGridInterpolator

#############################################################################
# Convenient Definitions                                                    #
#############################################################################

inds  = {'q': 0, 'qq': 1, 'B': 2, 'bq': 0, 'bqq': 1, 'bB': 2, 'g': 3}
process_ind = {'qbq_gg': 0, 'qq_qqg': 1, 'qqq_Bg': 2, 'other': 3}
def process_group(process):
    if (process == 'qbq_gg') | (process == 'qq_qqg') | (process == 'qqq_Bg'):
        return process
    else:
        return 'other'

#Define the interactions to be included in each equation
boltz_coll_list = {
    'q' :[
        ['q','bq','g','g'], ['q','bqq','bq','g'], ['q','bB','bqq','g'], ['q','bB','bq','bq'],
        ['q','q','qq','g'], ['q','qq','B','g'], ['q','B','qq','qq'],
        ['B','bqq','q','g'], ['qq','bq','q','g'], ['B','bq','q','q'],
        ['B','bB','q','bq'], ['qq','bqq','q','bq'], ['qq','bB','q','bqq']],  
    'qq':[
        ['qq','bq','q','g'], ['qq','bqq','g','g'], ['qq','bqq','q','bq'], ['q','bB','bq','g'],
        ['qq','q','B','g'], ['q','B','qq','qq'], 
        ['q','q','qq','g'],
        ['B','bq','qq','g'], ['B','bB','qq','bqq'], ['bB','qq','bqq','q'],
        ['B','bq','qq','bq']],
    'B':[
        ['B','bq','qq','g'], ['B','bq','q','q'], ['B','bqq','q','g'], ['B','bqq','qq','bq'],
        ['B','bB','g','g'], ['B','bB','q','bq'], ['B','bB','qq','bqq'],
        ['q','qq','B','g'], ['q','B','qq','qq']]
}

boltz_weights = {'q':[-1, -1, -1, -1, 
                -2, -1, -1, 
                1, 1, 1, 
                1, 1, 1],
            'qq':[
                -1, -1, -1, -1, 
                -1, 2, 1, 
                1, 1, -1, 
                1],
            'B':[
                -1, -1, -1, -1, 
                -1, -1, -1, 
                1, -1]
}



#############################################################################
#############################################################################
# SU(N) information                                                         #
#############################################################################
#############################################################################

# See http://pdg.lbl.gov/2009/reviews/rpp2009-rev-qcd.pdf
pouya_check = False

# LL coefficient for beta function
def beta_0(C_A, Nf=1, Tr=0.5):
    return (11*C_A-4*Nf*Tr)/(12*np.pi)

# Running coupling
def alpha_s(Q, LambdaD, Nc, Nf=1, Tr=0.5):
    return 6*np.pi/(11*Nc-2*Nf)/np.log(Q/LambdaD)
    #t = 2*np.log(Q/LambdaD)
    #C_A = Nc
    #return 1/(beta_0(C_A, Nf, Tr) * t)

# alpha evaluated at the Bohr radius of a bound state with reduced mass mu_r
def alpha_rBohr(mu_r, LambdaD):
    alph = 1
    for i in np.arange(6):
        alph = alpha_s(alph*mu_r, LambdaD, Nc=3, Nf=1, Tr=0.5)
    return alph


####################
# Bound State Info #
####################

# Binding Energies
def bindE(mq, LambdaD, hadron):
    alpha = alpha_rBohr(mq, LambdaD)
    if   (hadron == 'B') | (hadron == 'bB'):
        return -.26 * (4/3)**2 * alpha**2 * mq

    elif (hadron == 'qq') | (hadron == 'bqq') | (hadron == 'pi'):
        return -1/9 * alpha**2 * mq

    elif (hadron == 'q') | (hadron == 'bq'):
        return 0

    else:
        raise TypeError('Not a valid bound state')

# Total Energies (in units of LambdaD)
def mL(mq, LambdaD, hadron):

    #Number of constituent quarks
    if   (hadron == 'B') | (hadron == 'bB'):
        nq = 3
    elif (hadron == 'qq') | (hadron == 'bqq') | (hadron == 'pi'):
        nq = 2
    elif (hadron == 'g'):
        return 0
    else:
        nq = 1

    return (nq*mq + bindE(mq, LambdaD, hadron))/LambdaD + nq-1

def engL(mq, LambdaD, hadron):
    # Account for kinetic energy
    return mL(mq, LambdaD, hadron) + 1

# Degeneracy Factors
gs = {'q': 2*3*7/8, 'qq': 3, 'B': 3*7/8, 'pi': 1,
      'bq': 2*3*7/8, 'bqq': 3, 'bB': 3*7/8}

# Cross-Sections

# Load from Juri/Pouya's notebook

log10special_xsecs = {}
def special_xsecs_func(mq, LambdaD, process, reload_switch = False):
    log10Lam_range = np.arange(10,14+.05/2,.05)
    log10ratio_range = np.arange(2.0063,3.999+0.05,0.05)
    global log10special_xsecs

    if reload_switch:
        log10special_xsecs = {}
    if log10special_xsecs == {}:

        for string in ['qbq_gg', 'qq_qqg', 'qqq_Bg']:
            log10special_xsecs[string] = []
            #with open('/Users/gregoryridgway/Desktop/Composite DM/Data/xsec_'+string+'.csv') as csvfile:
            with open('/Users/gregoryridgway/Desktop/Composite DM/'+string+'.csv') as csvfile:
                reader = csv.reader(csvfile)
                reader = csv.reader(csvfile)
                for row in reader:
                    log10special_xsecs[string].append([np.log10(float(r)) for r in row])

            log10special_xsecs[string] = np.array(log10special_xsecs[string])
            #print(special_xsecs[string][0:2])

            if False:
                #print(log10Lam_range.shape, log10ratio_range.shape)
                #print(special_xsecs[string].shape)
                log10special_xsecs[string] = RegularGridInterpolator(
                        points = ( (log10Lam_range, log10ratio_range) ),
                        values = log10special_xsecs[string],
                        method = 'linear'
                        )
            else:
                log10special_xsecs[string] = interp1d(log10special_xsecs[string][:,0], log10special_xsecs[string][:,1])

    if False:
        log10ratio = np.log10(mq/LambdaD)
        log10LambdaD = np.log10(LambdaD)
        log10LambdaD_tmp, log10ratio_tmp = extrap(
                np.log10(LambdaD), log10ratio, 
                [log10Lam_range[0], log10Lam_range[-1], log10ratio_range[0], log10ratio_range[-1]]
                )
        return 10**(log10special_xsecs[process]( (log10LambdaD_tmp, log10ratio_tmp) ))
    else:
        log10ratio = np.log10(mq/LambdaD)
        log10ratio_tmp = extrap(log10ratio, bounds=[log10ratio_range[0], log10ratio_range[-1]])
        return 10**(log10special_xsecs[process](log10ratio_tmp))

def xsec(m, LambdaD, Nc, process):
    # Naive cross-section
    sigma_0 = np.pi*alpha_s(m, LambdaD, Nc)**2/m**2

    # Sommerfeld or geometrix enhancements
    special_process = (process == 'qbq_gg') | (process == 'qq_qqg') | (process == 'qqq_Bg')
    #special_process = False
    if special_process:
        # Nearest Neighbor extrapolation
        return special_xsecs_func(m, LambdaD, process) * sigma_0
    else:
        geometric_factor = alpha_rBohr(m, LambdaD)**-3
        return geometric_factor * sigma_0 /(4/3)

#############################################################################
#############################################################################
# Early Universe Thermodynamics                                             #
#############################################################################
#############################################################################

# Planck mass and reduced Planck Mass in eV
mpl=1.220910e19 * 1e9
Mpl=mpl/np.sqrt(8*np.pi)

gstar_data = []
gstar_func = None
def get_gstar(T, DM=False, LambdaD=None, Nc=None):
    """g_star as a function of T, including the contribution from DM or not

    Parameters
    ----------
    Returns
    -------
    """

    global gstar_data
    global gstar_func

    # Convert T to an array
    if isinstance(T*1.0, float):
        T = np.array([T])

    #Download the g_star data if it hasn't already been downloaded
    if (gstar_data == []) or (gstar_func == None):
        with open('/Users/gregoryridgway/Desktop/Webplot Digitizer/Classic_Results/g_star.csv') as csvfile:
            reader = csv.reader(csvfile)
            reader = csv.reader(csvfile)
            for row in reader:
                gstar_data.append([float(row[0])*1e9,float(row[1])])
        gstar_data = np.array(gstar_data)

        gstar_func = interp1d(
                gstar_data[:,0], gstar_data[:,1], 
                bounds_error=False, fill_value=(2,gstar_data[-1,1]), kind='linear'
                )

    gstar = gstar_func(T)
    if DM:
        gstar[T>LambdaD*1.01] = (gstar[T>=LambdaD*1.01] + (Nc**2-1))

    return gstar

def hubble(T, DM=True, LambdaD=None, Nc=None):
    return np.pi*T**2/Mpl * np.sqrt(get_gstar(T, DM=DM, LambdaD=LambdaD, Nc=Nc)/90)

# Entropy of the universe
def s(T, LambdaD, Nc, DM=True):
    g_star = get_gstar(T, DM=DM, LambdaD=LambdaD, Nc=Nc)
    return 2*np.pi**2/45 * g_star*T**3

# Number density of a particle
n_data = None
def n(T, m, g, part='fermion'):
    """
    set $x = T/m$, and $y = p/m$
    $$
    n = \frac{g T^3}{2 \pi^2}\int_0^\infty dy \frac{y^2}{e^{\sqrt{y^2+x^2}}\pm1}
    $$
    When x is small (smaller than $0.1$), we drop x from the above equation, and when it's large (larger than $10$) we drop the $\pm1$ to find

    $$
    \begin{align}
    n & = \alpha g \frac{\zeta(3) T^3}{\pi^2},\text{ for x<0.1} \\
    n & = g \left(\frac{m T}{2 \pi}\right)^{\frac{3}{2}} e^{-x},\text{ for x>10}
    \end{align}
    $$
    where $\alpha = 1 (3/4)$ for bosons (fermions). For all other values of x we make a 1D plot along x that we interpolate over.
    """

    # Numerically integrate the region between ultrarelativistic and non-relativistic
    global n_data
    if n_data == None:
        def integrand(y, x, sign):
            return y**2/(np.exp(np.sqrt(y**2+x**2)) + sign)
        inc=.005
        xlist = 10**np.arange(np.log10(.01),np.log10(50)+inc,inc)
        n_data = {
            'fermion': interp1d(np.log10(xlist),[np.log10(quad(integrand, 0, 50, args=(x,1))[0]) for x in xlist]),
            'boson': interp1d(np.log10(xlist),[np.log10(quad(integrand, 0, 50, args=(x,-1))[0]) for x in xlist])
        }

    # Non-relativistic Limit
    if isinstance(T*1.0, float):
        T = np.array([T])

    x = m/T
    output = np.zeros_like(x)
    mask = x>50
    output[mask] = g*(m * T[mask]/2/np.pi)**(3/2) * np.exp(-x[mask])
    if pouya_check:
        return g*(m * T/(2*np.pi))**(3/2) * np.exp(-x)

    if part == 'fermion':
        alpha=3/4
    else:
        alpha=1

    # Relativistic Limit
    mask = x<.01
    output[mask] = alpha * g * special.zeta(3) * T[mask]**3/np.pi**2


    # Intermediate Region
    mask = (x>.01) & (x<50)
    output[mask] = g * T[mask]**3 * 10**n_data[part](np.log10(x[mask]))/(2*np.pi**2)

    return output

# Yeq in the absence of a chemical potential
def Yeq(gDM, mDM, x, LambdaD, Nc, part='fermion'):
    T = mDM/x
    return n(T, mDM, gDM, part)/s(T, LambdaD, Nc)

# Boltzmann Equation for 
def get_history(init_cond, lnx_vec, mDM, sigmav, LambdaD, Nc, gDM, rtol=1e-6):
    """Returns...

    Parameters
    ----------
    Returns
    -------
    Note
    ----
    Following 1104.5548, eqn. 11 or 12

    """

    def lam(gstar):
        return 4*np.pi*mDM*Mpl*np.sqrt(gstar/90)

    def lnY_diff_eq(var, lnx):

        def dlnY_dlnx(lnY, lnx):
            x = np.exp(lnx)
            T = mDM/x
            gstar = get_gstar(T, DM=True, LambdaD=LambdaD, Nc=Nc)[0]

            return -(lam(gstar)*sigmav/x) * (np.exp(lnY) - Yeq(gDM, mDM, x, LambdaD, Nc)[0]**2 * np.exp(-lnY))

        lnY = var[0]

        return dlnY_dlnx(lnY, lnx)

    soln = odeint(lnY_diff_eq, init_cond, lnx_vec, rtol=rtol)

    return np.exp(soln[:,0])


#############################################################################
# Bubbles                                                                   #
#############################################################################

#############################################################################
# Pocket                                                                    #
#############################################################################
def vw(R, LambdaD, Nc):
    vterm=1e-3
    if False:
        Hc = hubble(LambdaD, True, LambdaD, Nc)
        R_1 = (100*LambdaD*Hc**2)**(-1/3)
        if True:
            vs = 100/3 * Hc*R_1 * (R_1/R)**2
        else:
            vs = 100/3 * Hc * R

        if isinstance(vs, float):
            vs = np.array([vs])

        vs[vs>1] = 1.

        return np.squeeze(vs)
    else:
        return vterm

    #return vs

def make_log_eq_fac_list(engLs, mLs):
    states = ['q', 'qq', 'B']
    eq_fac_list = {state: np.array([]) for state in states}
    for state in states: 
        for coll in boltz_coll_list[state]:
            if (coll[2] == 'g') & (coll[3] == 'g'):
                log_eq_fac = (
                    -engLs[coll[0]]-engLs[coll[1]] +
                    2*np.log(4*np.pi/3) +
                    -np.log(4 * np.pi**2) + 3/2*np.log(mLs[coll[0]]) +
                    3/2*np.log(mLs[coll[1]]) + np.log(gs[coll[0]]*gs[coll[1]])
                )
            elif (coll[3] == 'g'):
                log_eq_fac = (
                    engLs[coll[2]]-engLs[coll[0]]-engLs[coll[1]] +
                    np.log(4*np.pi/3) +
                    -3/2*np.log(2 * np.pi) + 3/2*np.log(mLs[coll[0]]) +
                    3/2*np.log(mLs[coll[1]]) - 3/2*np.log(mLs[coll[2]]) +
                    np.log(gs[coll[0]]*gs[coll[1]]/gs[coll[2]])
                )
            else:
                log_eq_fac = (
                    engLs[coll[2]]+engLs[coll[3]]-engLs[coll[0]]-engLs[coll[1]] +
                    3/2*np.log(mLs[coll[0]]) + 3/2*np.log(mLs[coll[1]])
                    - 3/2*np.log(mLs[coll[2]]) - 3/2*np.log(mLs[coll[3]]) +
                    np.log(gs[coll[0]]*gs[coll[1]]/gs[coll[2]]/gs[coll[3]])
                )
            eq_fac_list[state] = np.append(eq_fac_list[state], log_eq_fac)
        eq_fac_list[state] = np.array(eq_fac_list[state])


    return eq_fac_list

def collision(coll, log_eq_fac, logNs, y, mq, LambdaD, Nc, part, v_w, mLs, engLs, logxsec_list):

    # Add a fictitious gluon index
    logNgs = np.append(logNs, 0)

    #convert from y to R
    Rlam = np.exp(y)
    R = Rlam/LambdaD

    ## Ratio of equilibrium factors ##
    # e^(m3 + m4 - m1 - m2)/Tc * ()#
    if (coll[2] == 'g') & (coll[3] == 'g'):
        logR_fac = 6*np.log(Rlam)
        #log_eq_fac = (
        #    -engLs[coll[0]]-engLs[coll[1]] +
        #    2*np.log(4*np.pi/3) + 2*3*np.log(Rlam) +
        #    -np.log(4 * np.pi**2) + 3/2*np.log(mLs[coll[0]]) +
        #    3/2*np.log(mLs[coll[1]]) + np.log(gs[coll[0]]*gs[coll[1]])
        #)
    elif (coll[3] == 'g'):
        logR_fac = 3*np.log(Rlam)
        #log_eq_fac = (
        #    engLs[coll[2]]-engLs[coll[0]]-engLs[coll[1]] +
        #    np.log(4*np.pi/3) + 3*np.log(Rlam) +
        #    -3/2*np.log(2 * np.pi) + 3/2*np.log(mLs[coll[0]]) +
        #    3/2*np.log(mLs[coll[1]]) - 3/2*np.log(mLs[coll[2]]) +
        #    np.log(gs[coll[0]]*gs[coll[1]]/gs[coll[2]])
        #)
    else:
        logR_fac=0
        #log_eq_fac = (
        #    engLs[coll[2]]+engLs[coll[3]]-engLs[coll[0]]-engLs[coll[1]] +
        #    3/2*np.log(mLs[coll[0]]) + 3/2*np.log(mLs[coll[1]])
        #    - 3/2*np.log(mLs[coll[2]]) - 3/2*np.log(mLs[coll[3]]) +
        #    np.log(gs[coll[0]]*gs[coll[1]]/gs[coll[2]]/gs[coll[3]])
        #)

    ## sigmav/(4 pi R^3/3)^2 * (v_w / (4pi R^3/3))^{-1} * R/N_part ##
    process = coll[0]+coll[1]+'_'+coll[2]+coll[3]
    log_prefac = (
        - np.log(4*np.pi/3) - np.log(v_w) - 2*np.log(R)
        + logxsec_list[process_group(process)] - logNs[inds[part]]
    )

    #forward and backward processes
    forward  = log_prefac + logNgs[inds[coll[0]]] + logNgs[inds[coll[1]]]
    backward = log_prefac + logNgs[inds[coll[2]]] + logNgs[inds[coll[3]]] + log_eq_fac+logR_fac

    return -np.exp(forward) + np.exp(backward)


def get_B(y_vec, init, mq, LambdaD, Nc, vterm=1e-3, collision_list=boltz_coll_list, weights=boltz_weights, mxstep = 1000, rtol=1e-4, pion=False):
    """Calculates the abundance of baryons that survive the compression of the pocket

    Parameters
    ----------
    init : ndarray
        initial quark number in pocket
    LambdaD : float
        Confining scale for dark quarks
    mxstep : int, optional
        The maximum number of steps allowed for each integration point. See *scipy.integrate.odeint* for more information.
    rtol : float, optional
        The relative error of the solution. See *scipy.integrate.odeint* for more information.

    """

    engLs = {part: engL(mq, LambdaD, part) for part in inds}
    mLs   = {part: mL(mq, LambdaD, part) for part in inds}
    logxsec_list = {process: np.log(xsec(mq, LambdaD, Nc, process)) for process in process_ind}
    eq_fac_list = make_log_eq_fac_list(engLs, mLs)

    Nq_init = np.exp(init[0])
    Hc = hubble(LambdaD, True, LambdaD, Nc)

    # Define derivatives of lnN with respect to y = ln(R LambdaD)
    def derivs(y, var):
        R = np.exp(y)/LambdaD
        logNq, logNqq, logNB = var[0], var[1], var[2]
        logNs = np.array([logNq, logNqq, logNB])

        if 25*np.exp(logNq-2*y)/(4*np.pi/3) < 1 or True:
            v_w = vterm#vw(R, LambdaD, Nc, vterm=vterm)
        else:
            v_w = 1/2 * sum([weight*collision(coll, logNs, y, mq, LambdaD, Nc, 'q', 1, mLs, engLs, logxsec_list)
                       for coll, weight in zip(collision_list['q'], weights['q'])])
            #goods = [weight*collision(coll, logNs, y, mq, LambdaD, Nc, 'qq', v_w, mLs, engLs, logxsec_list)
            #           for coll, weight in zip(collision_list['qq'], weights['qq'])]
            #for coll, good in zip(collision_list['qq'], goods):
            #    print(coll, good)
            #sys.exit()

        # baryon velocity
        if pouya_check:
            v_b = np.sqrt(1/mL(mq, LambdaD, 'B'))
        else:
            v_b = np.sqrt(2/mL(mq, LambdaD, 'B'))

        if pion:
            # pion velocity
            v_pi = np.sqrt(2/mL(mq, LambdaD, 'qq'))


        # Derivatives
        def dlogNq_dy(y):
            return sum([weight*collision(coll, eq, logNs, y, mq, LambdaD, Nc, 'q', v_w, mLs, engLs, logxsec_list)
                       for eq, coll, weight in zip(eq_fac_list['q'], collision_list['q'], weights['q'])])

        def dlogNqq_dy(y):
            #goods = [weight*collision(coll, logNs, y, mq, LambdaD, Nc, 'qq', v_w, mLs, engLs, logxsec_list)
            #           for coll, weight in zip(collision_list['qq'], weights['qq'])]
            #for coll, good in zip(collision_list['qq'], goods):
            #    print(coll, good)
            #sys.exit()
            return sum([weight*collision(coll, eq, logNs, y, mq, LambdaD, Nc, 'qq', v_w, mLs, engLs, logxsec_list)
                       for eq, coll, weight in zip(eq_fac_list['qq'], collision_list['qq'], weights['qq'])])

        def dlogNB_dy(y):
            esc_term = 3/2 * v_b/v_w

            #goods = [weight*collision(coll, logNs, y, mq, LambdaD, Nc, 'B', v_w, mLs, engLs, logxsec_list)
            #           for coll, weight in zip(collision_list['B'], weights['B'])]
            #for coll, good in zip(collision_list['B'], goods):
            #    print(coll, good)
            #sys.exit()
            return sum([weight*collision(coll, eq, logNs, y, mq, LambdaD, Nc, 'B', v_w, mLs, engLs, logxsec_list)
                       for eq, coll, weight in zip(eq_fac_list['B'], collision_list['B'], weights['B'])
                        ]) + esc_term

        if pion:
            def dlogNpi_dy(logNq, logNqq, logNB, y):
                logNs = np.array([logNq, logNqq, logNB])
                esc_term = 3/2 * v_pi/v_w

                return sum([weight*collision(coll, logNs, y, mq, LambdaD, Nc, 'B', v_w, mLs, engLs, logxsec_list)
                           for coll, weight in zip(collision_list['B'], weights['B'])
                            ]) + esc_term

            logNq, logNqq, logNB, logNpi = var[0], var[1], var[2], var[3]

            return np.squeeze([
                dlogNq_dy( logNq, logNqq, logNB, y),
                dlogNqq_dy(logNq, logNqq, logNB, y),
                dlogNB_dy( logNq, logNqq, logNB, y),
                dlogNpi_dy(logNq, logNqq, logNB, y)
            ])

        else:

            return np.squeeze([dlogNq_dy(y),dlogNqq_dy(y),dlogNB_dy(y)])

    #print(derivs(y_vec[0], init))
    #raise TypeError('cross check')

    if False:
        return rk(
            derivs, init, 
            y_vec[0], y_vec[-1], step=y_vec[1]-y_vec[0], vec=y_vec, 
            rtol=rtol
        )
    else:
        return odeint(
            derivs, init, y_vec,
            mxstep = mxstep, tfirst=True, rtol=rtol
        )

def get_init(mq, LambdaD, Nc, Nq_init, y_init):
    # In the beginning the only important interaction for the diquark is q+q -> qq+g.  
    # Set the forward process equal to the backwards process to get this estimate.
    engLs = {part: engL(mq, LambdaD, part) for part in inds}
    mLs   = {part: mL(mq, LambdaD, part) for part in inds}
    log_eq_fac_qq = (
        engLs['qq']-2*engLs['q'] + np.log(4*np.pi/3) +
        -3/2*np.log(2 * np.pi) + 3*np.log(mLs['q']) - 3/2*np.log(mLs['qq']) +
        np.log(gs['q']**2/gs['qq'])
    )
    N2 = Nq_init**2/np.exp(log_eq_fac_qq+3*y_init)


    # For Baryons, it's q+qq -> B + g
    log_eq_fac_B = (
        engLs['B']-engLs['qq']-engLs['q'] + np.log(4*np.pi/3) +
        -3/2*np.log(2 * np.pi) + 3/2*np.log(mLs['q']) + 3/2*np.log(mLs['qq']) 
        - 3/2*np.log(mLs['B']) + np.log(gs['q']*gs['qq']/gs['B'])
    ) 
    N3 = Nq_init*N2/np.exp(log_eq_fac_B+3*y_init)

    return np.squeeze([np.log(Nq_init), np.log(N2), np.log(N3)])

def get_simple_Ns(collision_list, weights, Nq, y, mq, LambdaD, Nc, v_w, mLs, engLs, logxsec_list):
    # Check to see if the diquark and baryon bolt_eqs are dominated by q+q -> qq+g and q+qq -> B+g, respectively.
    # If so, set those terms to zero to get N_qq and N_B as functions of N_q
    # Remember to use dimensions of Lambda.

    R = np.exp(y)
    eq_fac_list = make_log_eq_fac_list(engLs, mLs)

    # Calculate the analytic expressions for qq and B #
    log_eq_fac_qq = (
        engLs['qq']-2*engLs['q'] +
        np.log(4*np.pi/3) +
        -3/2*np.log(2 * np.pi) + 3*np.log(mLs['q']) - 3/2*np.log(mLs['qq']) +
        np.log(gs['q']**2/gs['qq'])
    )

    log_eq_fac_B = (
        engLs['B']-engLs['qq']-engLs['q'] +
        np.log(4*np.pi/3) +
        -3/2*np.log(2 * np.pi) + 3/2*np.log(mLs['q']) + 3/2*np.log(mLs['qq'])
        - 3/2*np.log(mLs['B']) +
        np.log(gs['q']*gs['qq']/gs['B'])
    )

    N2 = Nq**2/np.exp(log_eq_fac_qq+3*y)
    N3 = N2*Nq/np.exp(log_eq_fac_B+3*y)
    logNs = np.log([Nq, N2, N3])

    good_approx=True
    if False:
        # Check to see if this was a good approximation #
        logNs_perturbed = np.log([Nq, N2*.99, N3*.99])
        rates = {state : [weight * collision(coll, logNs_perturbed, y,
            mq, LambdaD, Nc, state, v_w, mLs, engLs, logxsec_list)
            for coll, weight in zip(collision_list[state], weights[state])
        ] for state in ['qq', 'B']}
        
        #state='B'
        #for ii, rate in enumerate(rates[state]):
        #    print(collision_list[state][ii], rate)

        qq_ind = collision_list['qq'].index(['q', 'q', 'qq', 'g'])
        B_ind  = collision_list['B' ].index(['q', 'qq', 'B', 'g'])
        
        # Good means that for both bound states, the dominant rate is dominant by more than 90%
        qq_frac = np.abs(rates['qq'][qq_ind]/np.sum(np.abs(rates['qq'])))
        B_frac  = np.abs((rates['B'][B_ind]/np.sum(np.abs(rates['B']))))
        #print(B_frac)
        if (qq_frac < .9) or (B_frac < .9):

            good_approx=False
            #print('Not dominant')

        # Also, the total rate must be greater than 1
        elif (sum(rates['qq'])<1) or (sum(rates['B'])<1):
            good_approx=False
            print('No net change')

        else:
            good_approx=True


    dNq_dt = -Nq/R * sum([weight*collision(coll, eq, logNs, y, mq, LambdaD, Nc, 'q', 1, mLs, engLs, logxsec_list)
                for eq, coll, weight in zip(eq_fac_list['q'], collision_list['q'], weights['q'])])

    return good_approx, dNq_dt, logNs

#############################################################################
# Bubble Expansion Integration                                              #
#############################################################################

def expansion(init, LambdaD, Nc, v_0):
    """
    """

    #Set to True when nucleation was once efficient
    nuc_switch = False

    #When epsilon is smaller than this, nucleation is extremely inefficient
    eps_min = 1.1e-3

    #Adiabatic Cooling scale
    adia = hubble(LambdaD, True, LambdaD, Nc) / LambdaD

    def v_w(xi):
        return v_0
    
    def Gamma_c(eps, x):
        if eps<eps_min:
            return 0
        pre = 32*np.pi/3
        exp = np.exp(-0.7/(100*eps)**2)/(100*eps)**3
        return pre*exp*(1-x)

    def Gamma_w(eps, x):
        return 100*eps/2 * v_w(x)

    def add_Delta(eps_func, a, b):
        # Gaussian Quadrature
        step = (b-a)/2
        mid = (b+a)/2
        p1 = mid - 1/np.sqrt(3)
        p2 = mid + 1/np.sqrt(3)
        f1 = eps_func(p1)*v_w(p1)
        f2 = eps_func(p2)*v_w(p2)

        return step * (f1 + f2)

    # Define derivatives of eps and x with respect to xi = t*T_c
    def derivs(xi, var, nuc_switch):

        # Derivatives
        def dx_dxi(x, eps, xi):
            gc = Gamma_c(eps[-1], x[-1])

            if not nuc_switch:
                growth_term = 0
            else:
                growth_term = 0
            return gc + growth_term

        def deps_dxi(x,eps,xi):
            return adia - dx_dxi(x, eps, xi)/100


        x, eps = var[0], var[1]

        return np.squeeze([
            dx_dxi(x, eps, xi),
            deps_dxi(x,eps,xi)
        ])

    soln = the_loop()
    #soln = odeint(
    #    derivs, init, y_vec,
    #    mxstep = mxstep, tfirst=True, rtol=rtol
    #)
    return soln

def the_loop():
    return 0

#############################################################################
# Convenient Functions                                                      #
#############################################################################

def extrap(x_list, y_list=None, bounds=None):
    """ A function to be used with RegularGridInterpolator to make nearest-neighbor extrapolation easy
    
    bounds = np.array([x_low, x_high, y_low, y_high])
    """

    ## extrapolation: x
    #if (dim == 'x') | (dim == 'both'):
    if isinstance(x_list*1.0, float):
        x_list = np.array([x_list])
    x_list[x_list>bounds[1]] = bounds[1]
    x_list[x_list<bounds[0]] = bounds[0]

    if y_list == None:
        return x_list
    else:
        ## extrapolation: y
        #if (dim == 'y') | (dim == 'both'):
        if isinstance(y_list*1.0, float):
            y_list = np.array([y_list])
        y_list[y_list>bounds[3]] = bounds[3]
        y_list[y_list<bounds[2]] = bounds[2]

        return np.squeeze(x_list), np.squeeze(y_list)

#def integrate(init, derivs, xi, xf, method='Euler'):
#    """
#    """
#    if method == 'Euler':

def rk(deriv, init, t0, t_end, step, vec, rtol=0.001, atol=1e-6):

    # Constants -- Cash-Karp
    ca  = np.array([0, 1./5, 3./10, 3./5, 1, 7./8])
    cb  = np.array([
        [0, 0, 0, 0, 0],
        [1./5, 0, 0, 0, 0],
        [3./40, 9./40, 0, 0, 0],
        [3./10, -9./10, 6./5, 0, 0],
        [-11./54, 5./2, -70./27, 35./27, 0],
        [1631./55296, 175./512, 575./13824, 44275./110592, 253./4096]
    ])
    cc  = np.array([37./378, 0, 250./621, 125./594, 0, 512./1771])
    ccs = np.array([2825./27648, 0, 18575./48384, 13525./55296, 277./14336, 1./4])

    k   = np.zeros((6,init.size))
    yp,yn,ys  = np.zeros(6),np.zeros(6),np.zeros(6)

    # Initialize the loop
    y     = init
    t_cur = t0

    ylist = [init]
    tlist = [t_cur]

    sign = np.sign(t_end-t_cur)
    while sign*t_cur < sign*t_end:
        # Let the last step take you just to t_end
        if sign*(t_cur+step) > sign*t_end:
            step=t_end-t_cur
        for i in np.arange(6):
            yp = y+np.dot(cb[i],k[:5])
            k[i] = step*deriv(t_cur+ca[i]*step, yp);

        ys = y + np.dot(ccs,k)
        yn = y + np.dot(cc,k)

        #constraints on relative and absolute error
        delta = np.sum(np.sqrt(np.abs(ys-yn)))
        norm = np.sum(np.sqrt(np.abs(yn)))
        if norm>0:
            Delta = np.sum(np.sqrt(np.abs(ys-yn)))/norm
        else:
            Delta = 0

        if(Delta>delta):
            delta=Delta

        if delta == 0:
            delta = rtol

        if(delta > rtol):
            step = 0.9*step*(rtol/delta)**0.2
        else:
            y = yn
            t_cur += step;

            tlist.append(t_cur)
            ylist.append(y)

            # adjust next step
            dtau = 0.9*step*(rtol/delta)**0.2
            if(step>0.05):
                step=0.05

    #if vec != None:
    #    return vec, interp1d(np.array(tlist),np.array(ylist))(vec)
    #else:
    return np.array(tlist), np.array(ylist)

#############################################################################
# Graveyard of Code I can't Bear to Erase                                   #
#############################################################################
def OLDget_init(collision_list, weights, mq, LambdaD, Nc, Nq_init, y_init, y_final, init=None, step=.01, mxstep = 10000, rtol=1e-4, pion=False):
    if True:
        engLs = {part: engL(mq, LambdaD, part) for part in inds}
        mLs   = {part: mL(mq, LambdaD, part) for part in inds}
        log_eq_fac_qq = (
            engLs['qq']-2*engLs['q'] + np.log(4*np.pi/3) +
            -3/2*np.log(2 * np.pi) + 3*np.log(mLs['q']) - 3/2*np.log(mLs['qq']) +
            np.log(gs['q']**2/gs['qq'])
        )
        N2 = Nq_init**2/np.exp(log_eq_fac_qq+3*y_init)

        log_eq_fac_B = (
            engLs['B']-engLs['qq']-engLs['q'] + np.log(4*np.pi/3) +
            -3/2*np.log(2 * np.pi) + 3/2*np.log(mLs['q']) + 3/2*np.log(mLs['qq']) 
            - 3/2*np.log(mLs['B']) + np.log(gs['q']*gs['qq']/gs['B'])
        ) 
        N3 = Nq_init*N2/np.exp(log_eq_fac_B+3*y_init)

        return np.squeeze([np.log(Nq_init), np.log(N2), np.log(N3)])
    else:
        y_vec = np.arange(y_init, y_final, -step)
        if y_vec[-1] != y_final:
            y_vec = np.append(y_vec, y_final)

        if init is None:
            init = [np.log(Nq_init), 3.74315348, -32.70274368]

        pre_soln = get_B(y_vec, init, collision_list, weights, mq, LambdaD, Nc, mxstep = mxstep, rtol=rtol, pion = False)


        # Second run
        y_vec = np.arange(y_final+step, y_final, -step)
        init = pre_soln[-2]
        init[0]=np.log(Nq_init)
        pre_soln = get_B(y_vec, init, collision_list, weights, mq, LambdaD, Nc, mxstep = mxstep, rtol=rtol, pion = False)
        #if np.any(np.isnan(pre_soln[-1])):
        #    raise TypeError('initial condition is nan')

        return pre_soln[-1]
