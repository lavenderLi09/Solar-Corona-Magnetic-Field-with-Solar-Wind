'''
Code       : pfss_script.py
Date       : 2024.10.01
Contributer: G.Y.Chen(gychen@smail.nju.edu.cn)
Purpose    : Compute the Potential Field Source Surface model according to the given HMI synoptic map...

### --------------------------------- ###
Remark:
2024.10.01: Build the code (gychen@smail.nju.edu.cn)
2024.10.08: Add the advanced fast mode to build field.
'''

import pickle
import matplotlib.pyplot as plt
import sunpy
import sunpy.map
import time
import os
import multiprocessing
import math
import torch
import numpy as np
import argparse
import vtk
import glob

from astropy.io import fits
from scipy.interpolate import griddata
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.special import sph_harm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import lpmv
from matplotlib.colors import LogNorm, PowerNorm
from multiprocessing import Manager
from pyevtk.hl import gridToVTK

# ============================

def compute_alm_blm(l, m, Rs,
                    progress_list, total_tasks, lock,print_interval=100, t0=0
                   ):
    Blm = np.sum(Br_resample*np.conj(sph_harm(m, l, PP, TT))*np.sin(TT)*dth*dph)/(l*Rs**(-2*l-1)+l+1)
    # Blm = np.sum(Br_resample*np.conj(Spherical_Harmonics(l, m, TT, PP))*np.sin(TT)*dth*dph)/(l*Rs**(-2*l-1)+l+1)
    Alm = - Blm * Rs ** (-2 * l - 1)

    with lock:
        progress_list[0] += 1 
        ti = time.time()
        if progress_list[0] % print_interval == 0 or progress_list[0] == total_tasks or progress_list[0]==1:
            print(f"Progress: {progress_list[0]:6d}/{total_tasks} tasks completed.  Wall_time: {(ti-t0)/60:6.3f} min", flush=True)
    return l, m, Alm, Blm

def parallel_pfss_coefficients(lmax=120, n_cores=10, print_interval=100, Rs=2.5):
    print('### =========== Computing Alm Blm  ============ ###')
    print(f'#       Available CPU cores: {multiprocessing.cpu_count():3d}                  #')
    print(f'#            Used CPU cores: {n_cores:3d}                  #')
    print('### =========================================== ###')
    t0 = time.time()
    manager = Manager()
    progress_list = manager.list([0])
    lock = manager.Lock() 
    lm_list = [[il, im] for il in range(lmax + 1) for im in range(-il, il + 1)]
    total_tasks = len(lm_list)
    Alm = {il: {} for il in range(lmax + 1)}
    Blm = {il: {} for il in range(lmax + 1)}
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = []
        for il,im in lm_list:
            futures.append(executor.submit(compute_alm_blm, il, im, Rs, progress_list, total_tasks, lock, print_interval, t0))
        for future in as_completed(futures):
            l,m,alm,blm = future.result()
            Alm[l][m]=alm
            Blm[l][m]=blm

    return Alm,Blm

def double_factorial(n):
    arr = range(n,0,-2)
    return math.prod(arr)

def Associated_Legendre(l,m,x, **kwargs):
    """
    Calculate the associated Legendre polynomial P_l^m.
    
    Parameters:
    l (int): degree of the polynomial
    m (int): order of the polynomial
    x (array_like): points at which to evaluate the polynomial
    
    Returns:
    array_like: values of the polynomial at points x
    """
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    pn1    = kwargs.get('pn1', None)
    pn2    = kwargs.get('pn2', None)
    if pn1 is not None and pn2 is not None:
        plm = ((2*l-1)*x*pn1-(l+m-1)*pn2)/(l-m)
        return plm
    is_array = False
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
        is_array = True
    if np.abs(m)>l or torch.any(torch.abs(x)>1):
        raise ValueError("`m` cannot larget than `l`, or input |`x`| need to less than 1.")
    if m<0:
        m=-m
        plm = (-1)**m*math.factorial(l-m)/math.factorial(l+m)*Associated_Legendre(l,m,x,**kwargs)
    elif l==m:
        if l<=80:
            plm = (-1)**l*float(double_factorial(2*l-1))*(1-x**2)**(l/2)
        else:
            # print('test OK')
            plm = -(2*l-1)*torch.sqrt(1-x**2)*Associated_Legendre(l-1,l-1,x,**kwargs)
    elif l==m+1:
        return x*(2*m+1)*Associated_Legendre(m,m,x)
    else:
        pn2 = Associated_Legendre(m,m,x)
        pn1 = Associated_Legendre(m+1,m,x)
        for il in range(m+2,l+1):
            plm = ((2*il-1)*x*pn1-(il+m-1)*pn2)/(il-m)
            pn2,pn1 = pn1,plm

    if is_array:
        return plm.detach().cpu().numpy()
    else:
        return plm

def Spherical_Harmonics(l,m,theta,phi, **kwargs):
    is_array = False
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    if isinstance(theta, np.ndarray) or isinstance(phi, np.ndarray):
        theta = torch.from_numpy(theta).to(device)
        phi   = torch.from_numpy(phi).to(device)
        is_array = True
    Plm = kwargs.get('Plm', None)
    Plm = Associated_Legendre(l, m, torch.cos(theta)) if Plm is None else Plm
    factorial1 = math.factorial(l-m)
    factorial2 = math.factorial(l+m)
    digit1     = len(str(factorial1))
    digit2     = len(str(factorial2))
    digit0     = 300
    if digit1<digit0 and digit2<digit0:
        Ylm = np.sqrt((2*l+1)/(4*np.pi)*float(factorial1)/float(factorial2))*Plm*torch.exp(1j*m*phi)
    elif digit1<digit0 and digit2>=digit0:
        Ylm = np.sqrt((2*l+1)/(4*np.pi)*float(factorial1)/float(factorial2/10**digit2))*Plm*torch.exp(1j*m*phi)
        for i in range(int(np.ceil(digit2/digit0))):
            # print(f'digit2={digit2}')
            Ylm = Ylm/np.sqrt(float(10**min(digit0, digit2)))
            digit2=digit2-digit0
    elif digit1>=digit0 and digit2<digit0:
        Ylm = np.sqrt((2*l+1)/(4*np.pi)*float(factorial1/10**digit1)/float(factorial2))*Plm*torch.exp(1j*m*phi)
        for i in range(int(np.ceil(digit1/digit0))):
            Ylm = Ylm*np.sqrt(float(10**min(digit0, digit1)))
            digit1=digit1-digit0
    else:
        Ylm = np.sqrt((2*l+1)/(4*np.pi)*float(factorial1/10**digit1)/float(factorial2/10**digit2))*Plm*torch.exp(1j*m*phi)
        for i in range(int(np.ceil(digit2/digit0))):
            Ylm = Ylm/np.sqrt(float(10**min(digit0, digit2)))
            digit2=digit2-digit0
        for i in range(int(np.ceil(digit1/digit0))):
            Ylm = Ylm*np.sqrt(float(10**min(digit0, digit1)))
            digit1=digit1-digit0
    if is_array:
        return Ylm.detach().cpu().numpy()
    else:
        return Ylm

def Brtp_lm(l,m, rr, tt, pp,**kwargs):
    P_l00 = kwargs.get('P_l00',None)
    P_lp1 = kwargs.get('P_lp1',None)
    is_array = False
    err    = kwargs.get('err', 1e-10)
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    if isinstance(tt, np.ndarray) or isinstance(pp, np.ndarray):
        tt = torch.from_numpy(tt).to(device)
        pp = torch.from_numpy(pp).to(device)
        rr = torch.from_numpy(rr).to(device)
        is_array = True
    Alm_lm = np.nan_to_num(Alm[l][m])
    Blm_lm = np.nan_to_num(Blm[l][m])
    if np.abs(Alm_lm)<err or np.abs(Blm_lm)<err:
        br_lm = torch.zeros_like(rr)
        bt_lm = torch.zeros_like(tt)
        bp_lm = torch.zeros_like(pp)
    else:
        if m>=0:
            Y_lm    = Spherical_Harmonics(l,  m,tt,pp,Plm=P_l00)
            Y_lp1_m = Spherical_Harmonics(l+1,m,tt,pp,Plm=P_lp1)
        else:
            Y_lm    = torch.conj(Spherical_Harmonics(l,  -m,tt,pp,Plm=P_l00))*(-1)**(-m)
            Y_lp1_m = torch.conj(Spherical_Harmonics(l+1,-m,tt,pp,Plm=P_lp1))*(-1)**(-m)
        br_lm  = (Alm_lm * l * torch.pow(rr, l - 1) - (l + 1) * Blm_lm * torch.pow(rr, -l - 2))*Y_lm
        bp_lm  = 1/torch.sin(tt)*(Alm_lm * torch.pow(rr, l-1) + Blm_lm * torch.pow(rr, -l - 2)) * 1j * m * Y_lm
        dY_dth = 1/torch.sin(tt)*(-(l+1)*Y_lm*torch.cos(tt)+(l-m+1)*Y_lp1_m*np.sqrt((2*l+1)*(l+m+1)/(2*l+3)/(l-m+1)))
        bt_lm  = (Alm_lm*torch.pow(rr,l-1)+Blm_lm*torch.pow(rr,-l-2))*dY_dth

    if is_array:
        br_lm = br_lm.detach().cpu().numpy()
        bt_lm = bt_lm.detach().cpu().numpy()
        bp_lm = bp_lm.detach().cpu().numpy()
    return br_lm,bt_lm,bp_lm

def brtp2bxyz(brtp,rtp=None,xyz=None):
    '''
    Convert the [Br, Bp, Bt] to [Bx, By, Bz] under the Cartesian meshgrid.
    '''
    br,bt,bp = brtp
    if rtp is not None:
        R,T,P=rtp
    elif xyz is not None:
        X, Y, Z  = xyz
        R = np.sqrt(X**2+Y**2+Z**2)
        T = np.arcsin(Z/R)
        P = np.arctan(Y/X)
        P = np.where(X<0,P+np.pi,P)
    else:
        raise ValueError('Grid point `rtp` in spherical system or `xyz` in cartesian system should be provide...')
    bx = br*np.cos(T)*np.cos(P)+bt*np.sin(T)*np.cos(P)-bp*np.sin(P)
    by = br*np.cos(T)*np.sin(P)+bt*np.sin(T)*np.sin(P)+bp*np.cos(P)
    bz = br*np.sin(T)-bt*np.cos(T)
    b_vec = np.stack([bx,by,bz])
    return b_vec

def task_lm(idx_i,idx_f,lm_list,total_tasks, print_interval, lock,progress_list,t0,err=1e-200):
    device='cpu'
    Br = torch.zeros_like(rr, dtype=torch.complex128).to(device)
    Bt = torch.zeros_like(tt, dtype=torch.complex128).to(device)
    Bp = torch.zeros_like(pp, dtype=torch.complex128).to(device)
    for l,m in lm_list[idx_i:idx_f]:
        br,bt,bp = Brtp_lm(l,m,rr,tt,pp,err=err,device=device)
        Br+=torch.nan_to_num(torch.real(br))
        Bt+=torch.nan_to_num(torch.real(bt))
        Bp+=torch.nan_to_num(torch.real(bp))
        with lock:
            progress_list[0]+=1
            if progress_list[0] % print_interval == 0 or progress_list[0] == total_tasks or progress_list[0]==1:
                ti = time.time()
                print(f"Progress: {progress_list[0]:6d}/{total_tasks} tasks completed.  Wall_time: {(ti-t0)/60:8.3f} min, "+\
                      f"max Br: {torch.real(Br).max().item():+.4e}, min Br:{torch.real(Br).min().item():+.4e}", flush=True)

    return Br,Bt,Bp

def parallel_build_Brtp(lmax=120, n_cores=10, print_interval=100, err=1e-200):
    print('### ============== Parallel PFSS ============== ###')
    print(f'#       Available CPU cores: {multiprocessing.cpu_count():3d}                  #')
    print(f'#            Used CPU cores: {n_cores:3d}                  #')
    print('### =========================================== ###')
    t0 = time.time()
    manager = Manager()
    progress_list = manager.list([0])  
    lock = manager.Lock() 
    lm_list = [[il, im] for il in range(lmax + 1) for im in range(-il, il + 1)]
    total_tasks = len(lm_list)
    chunks = total_tasks//n_cores
    device = torch.device('cpu')
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = []
        for i in range(n_cores):
            idx_i = i * chunks
            idx_f = (i + 1) * chunks if i < n_cores - 1 else total_tasks
            futures.append(executor.submit(task_lm, idx_i, idx_f, lm_list,total_tasks, print_interval, lock, progress_list, t0, err))

        Br = torch.zeros_like(rr, dtype=torch.complex128).to(device)
        Bt = torch.zeros_like(tt, dtype=torch.complex128).to(device)
        Bp = torch.zeros_like(pp, dtype=torch.complex128).to(device)
        for future in as_completed(futures):
            br,bt,bp = future.result()
            Br+=br
            Bp+=bp
            Bt+=bt
    Br = -torch.real(Br).cpu().numpy()
    Bt = -torch.real(Bt).cpu().numpy()
    Bp = -torch.real(Bp).cpu().numpy()
    print(f'Task finished. Total time: {(time.time()-t0)/60:8.3f} min', flush=True)
    return Br,Bt,Bp
    
# ============================
parser = argparse.ArgumentParser(description='Computing PFSS model')
parser.add_argument('-f'          , type=str    , help='Path to .fits file for synoptic map'       , required=False, default=''    )
parser.add_argument('-l'          , type=int    , help='lmax level for spherical harmonics'        , required=False, default=80    )
parser.add_argument('-nr'         , type=int    , help='Numbers of the r resampling'               , required=False, default=400   )
parser.add_argument('-nt'         , type=int    , help='Numbers of the theta resampling'           , required=False, default=200   )
parser.add_argument('-np'         , type=int    , help='Numbers of the phi resampling'             , required=False, default=400   )
parser.add_argument('-rs'         , type=float  , help='Source Surface radia'                      , required=False, default=2.5   )
parser.add_argument('--err'       , type=float  , help='The acceptable minimum error value'        , required=False, default=1e-200)
parser.add_argument('--n1_print'  , type=int    , help='The print interval when Alm,Blm computing' , required=False, default=100   )
parser.add_argument('--n2_print'  , type=int    , help='The print interval when Brtp computing'    , required=False, default=100   )
parser.add_argument('--n_cores'   , type=int    , help='Process cores numbers to paralle computing', required=False, default=10    )
parser.add_argument('--device'    , type=str    , help='Device for gpu when computing Brtp'        , required=False, default=None  )
parser.add_argument('--start'     , type=int    , help='Start index, default is 0'                 , required=False, default=0     )
parser.add_argument('--end'       , type=int    , help='End index, default is -1'                  , required=False, default=-1    )
parser.add_argument('--bound'     , type=str    , help='The .npy file for given boundary'          , required=False, default=None  )
parser.add_argument('--rtp'       , type=str    , help='The .npy file for given rtp'               , required=False, default=None  )
parser.add_argument('--save_vts'  , action='store_true' )
parser.add_argument('--save_vtu'  , action='store_true' )
parser.add_argument('--save_coef' , action='store_true' )
parser.add_argument('--load_coef' , action='store_true' )
parser.add_argument('--skip_Brtp' , action='store_true' )
parser.add_argument('--fast_mode' , action='store_true' )

args      = parser.parse_args()
fits_file = args.f
lmax      = args.l
n_r       = args.nr
n_t       = args.nt
n_p       = args.np
err       = args.err
save_coef = args.save_coef
n1_print  = args.n1_print
n2_print  = args.n2_print
n_cores   = args.n_cores
device    = args.device if args.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
idx_i     = args.start
idx_f     = args.end
load_coef = args.load_coef
save_vts  = args.save_vts
save_vtu  = args.save_vtu
skip_Brtp = args.skip_Brtp
fast_mode = args.fast_mode
Rs        = args.rs
BD        = args.bound
rtp_file  = args.rtp
t0        = time.time()

print(' ### ======================================================================= ### ')
print(' ###        Step1: Compute the Sphrical harmonics function coefficient       ### ')
print(' ### ======================================================================= ### ')

if BD is None:
    hmi_map = sunpy.map.Map(fits_file)
    deg   = np.pi/180
    nth,nph = hmi_map.data.shape
    sin_theta_ls = np.linspace(-1,1,nth)
    theta_ls     = np.arcsin(sin_theta_ls)
    phi_ls       = np.linspace(0,2*np.pi, nph)
    Th,Ph = np.meshgrid(theta_ls, phi_ls, indexing='ij')
    Br = hmi_map.data

    nt_resample = n_t
    np_resample = n_p
    dth = np.pi/nt_resample
    dph = np.pi/np_resample*2
    T_resample = np.linspace(-90*deg, 90*deg, nt_resample+1)[:nt_resample]+0.5*dth
    P_resample = np.linspace(0, 2*np.pi, np_resample+1)[:np_resample]+0.5*dph
    TT,PP  = np.meshgrid(T_resample, P_resample, indexing='ij')
    
    p_idx  = PP/(2*np.pi/(nph-1))
    t_idx  = (np.sin(TT)+1)/(2/(nth-1))
    p_size = p_idx.shape
    t_size = t_idx.shape
    p_idx  = p_idx.flatten()
    t_idx  = t_idx.flatten()
    p_idx0 = np.floor(p_idx).astype(int)
    p_idx1 = p_idx0+1
    t_idx0 = np.floor(t_idx).astype(int)
    t_idx1 = t_idx0+1
    Br_resample = Br[(t_idx0,p_idx0)]*(p_idx1-p_idx)*(t_idx1-t_idx)/(p_idx1-p_idx0)/(t_idx1-t_idx0)+\
                  Br[(t_idx1,p_idx0)]*(t_idx-t_idx0)*(p_idx1-p_idx)/(t_idx1-t_idx0)/(p_idx1-p_idx0)+\
                  Br[(t_idx0,p_idx1)]*(t_idx1-t_idx)*(p_idx-p_idx0)/(t_idx1-t_idx0)/(p_idx1-p_idx0)+\
                  Br[(t_idx1,p_idx1)]*(t_idx-t_idx0)*(p_idx-p_idx0)/(t_idx1-t_idx0)/(p_idx1-p_idx0)
    Br_resample = Br_resample.reshape(p_size)
else:
    print(f'Load Lower Boundary form {BD}...')
    deg   = np.pi/180
    Br    = np.load(BD)
    nth,nph = Br.shape
    theta_ls     = np.linspace(0,  np.pi, nth)
    phi_ls       = np.linspace(0,2*np.pi, nph)
    Th,Ph = np.meshgrid(theta_ls, phi_ls, indexing='ij')

    nt_resample = n_t
    np_resample = n_p
    dth = np.pi/nt_resample
    dph = np.pi/np_resample*2
    T_resample = np.linspace(-90*deg, 90*deg, nt_resample+1)[:nt_resample]+0.5*dth
    P_resample = np.linspace(0, 2*np.pi, np_resample+1)[:np_resample]+0.5*dph
    TT,PP  = np.meshgrid(T_resample, P_resample, indexing='ij')

    p_idx  = PP/(2*np.pi/(nph-1))
    t_idx  = (TT+90*deg)/np.pi*(nth-1)
    p_size = p_idx.shape
    t_size = t_idx.shape
    p_idx  = p_idx.flatten()
    t_idx  = t_idx.flatten()
    p_idx0 = np.floor(p_idx).astype(int)
    p_idx1 = p_idx0+1
    t_idx0 = np.floor(t_idx).astype(int)
    t_idx1 = t_idx0+1
    Br_resample = Br[(t_idx0,p_idx0)]*(p_idx1-p_idx)*(t_idx1-t_idx)/(p_idx1-p_idx0)/(t_idx1-t_idx0)+\
                  Br[(t_idx1,p_idx0)]*(t_idx-t_idx0)*(p_idx1-p_idx)/(t_idx1-t_idx0)/(p_idx1-p_idx0)+\
                  Br[(t_idx0,p_idx1)]*(t_idx1-t_idx)*(p_idx-p_idx0)/(t_idx1-t_idx0)/(p_idx1-p_idx0)+\
                  Br[(t_idx1,p_idx1)]*(t_idx-t_idx0)*(p_idx-p_idx0)/(t_idx1-t_idx0)/(p_idx1-p_idx0)
    Br_resample = Br_resample.reshape(p_size)
TT = 90*deg-TT

flag = False
if load_coef:
    with open('Alm.pkl', 'rb') as f:
        Alm = pickle.load(f)
    with open('Blm.pkl', 'rb') as f:
        Blm = pickle.load(f)
    if max(Alm.keys())<lmax:
        print('!! Warning: The coefficient files have the lmax less than the given one, try to calculate !!')
        flag = True
if not load_coef or flag:
    Alm, Blm = parallel_pfss_coefficients(lmax=lmax, print_interval=n1_print, n_cores=n_cores, Rs=Rs)
    if save_coef:
        with open('Alm.pkl', 'wb') as f:
            pickle.dump(Alm, f)
        with open('Blm.pkl', 'wb') as f:
            pickle.dump(Blm, f)

if not skip_Brtp:
    print(' ### ======================================================================= ### ')
    print(' ###                      Step2: Building the Brtp field                     ### ')
    print(' ### ======================================================================= ### ')
    print('Device: ', torch.device(device))
    if device != 'cpu':
        if rtp_file is None:
            nr = n_r
            r_list = np.linspace(1, Rs, nr)
            rr, tt, pp = np.meshgrid(r_list, T_resample, P_resample, indexing='ij')
            tt = 90*deg-tt
        else:
            rr,tt,pp = np.load(rtp_file)
        
        tt = torch.from_numpy(tt).type(torch.float64).to(device)
        pp = torch.from_numpy(pp).type(torch.float64).to(device)
        rr = torch.from_numpy(rr).type(torch.float64).to(device)
        
        Br = torch.zeros_like(rr, dtype=torch.complex128).to(device)
        Bt = torch.zeros_like(tt, dtype=torch.complex128).to(device)
        Bp = torch.zeros_like(pp, dtype=torch.complex128).to(device)
        t0 = time.time()
        icnt = 0
        if not fast_mode:
            lm_list = [[il, im] for il in range(lmax + 1) for im in range(-il, il + 1)]
            total_tasks=len(lm_list)
            print_interval=n2_print
            idx_f = idx_f if idx_f > 0 else total_tasks
            lm_list = lm_list[idx_i:idx_f]
            icnt=0
            is_traversal = (idx_i==0 and idx_f==total_tasks-1)
            if not is_traversal:
                print(f"Start Index: {idx_i: 6d};   End Index: {idx_f: 6d}")
            for l,m in lm_list:
                icnt+=1
                br,bt,bp = Brtp_lm(l,m,rr,tt,pp,err=err,device=device)
                Br+=br
                Bt+=bt
                Bp+=bp
                if (icnt % print_interval == 0 or icnt == total_tasks or icnt==1):
                    ti = time.time()
                    print(f"Progress: {icnt:6d}/{total_tasks} tasks completed.  Wall_time: {(ti-t0)/60:8.3f} min, "+\
                          f"max Br: {torch.real(Br).max().item():+.4e}, min Br:{torch.real(Br).min().item():+.4e}", flush=True)
        else:
            assign_tasks = idx_i!=0 or idx_f>0
            lm_list = [[il, im] for im in range(lmax + 1) for il in range(im, lmax+1)]
            total_tasks=len(lm_list) if not assign_tasks else int((idx_f-idx_i)*(2*lmax-idx_f-idx_i+3)/2)
            idx_f = idx_f if idx_f>0 else lmax+1
            for m in range(idx_i,idx_f):
                for l in range(m,lmax+1):
                    icnt+=1
                    if l==m:
                        P_l00 = Associated_Legendre(l,  m, torch.cos(tt))
                        P_lp1 = Associated_Legendre(l+1,m, torch.cos(tt))
                        br,bt,bp = Brtp_lm(l,m,rr,tt,pp,err=err,device=device,P_l00=P_l00,P_lp1=P_lp1)
                        Br+=br
                        Bt+=bt
                        Bp+=bp
                        if l!=0:
                            br,bt,bp = Brtp_lm(l,-m,rr,tt,pp,err=err,device=device,P_l00=P_l00,P_lp1=P_lp1)
                            Br+=br
                            Bt+=bt
                            Bp+=bp
                        P_ln1 = P_l00
                        P_l00 = P_lp1
                    else:
                        P_lp1 = Associated_Legendre(l+1, m, torch.cos(tt), pn2=P_ln1, pn1=P_l00)
                        br,bt,bp = Brtp_lm(l,m,rr,tt,pp,err=err,device=device,P_l00=P_l00,P_lp1=P_lp1)
                        Br+=br
                        Bt+=bt
                        Bp+=bp
                        if l!=0:
                            br,bt,bp = Brtp_lm(l,-m,rr,tt,pp,err=err,device=device,P_l00=P_l00,P_lp1=P_lp1)
                            Br+=br
                            Bt+=bt
                            Bp+=bp
                        P_ln1 = P_l00
                        P_l00 = P_lp1
            
                    if (icnt % n2_print == 0 or icnt == total_tasks or icnt==1):
                        ti = time.time()
                        print(f"Progress: {icnt:6d}/{total_tasks} tasks completed.  Wall_time: {(ti-t0)/60:8.3f} min, "+\
                              f"max Br: {torch.real(Br).max().item():+.4e}, min Br:{torch.real(Br).min().item():+.4e}", flush=True)
        
        Br = -np.real(Br.detach().cpu().numpy())
        Bt = -np.real(Bt.detach().cpu().numpy())
        Bp = -np.real(Bp.detach().cpu().numpy())
        
        if (idx_i==0 and idx_f==total_tasks-1) or (fast_mode and idx_i==0 and idx_f==lmax+1):
            np.savez('./Brtp.npz', **dict(Br=Br,Bt=Bt,Bp=Bp))
        else:
            os.makedirs('./Brtp_temp_files/', exist_ok=True)
            np.savez(f'./Brtp_temp_files/Brtp_{idx_i:06d}_{idx_f:06d}.npz', **dict(Br=Br, Bt=Bt, Bp=Bp))
            ti = time.time()
            print(f'Assigning Task: {idx_i:6d}  --{idx_f:6d}, Wall_time: {(ti-t0)/60:8.3f} min')
    else:
        nr = n_r
        r_list = np.linspace(1, Rs, nr)
        rr, tt, pp = np.meshgrid(r_list, T_resample, P_resample, indexing='ij')
        tt = 90*deg-tt
        
        tt = torch.from_numpy(tt).type(torch.float64)
        pp = torch.from_numpy(pp).type(torch.float64)
        rr = torch.from_numpy(rr).type(torch.float64)
        
        Br,Bt,Bp = parallel_build_Brtp(lmax=lmax, n_cores=n_cores, print_interval=n2_print, err=err)
        np.savez('./Brtp.npz', **dict(Br=Br,Bt=Bt,Bp=Bp))
        

if save_vts and not skip_Brtp:
    print(' ### ======================================================================= ### ')
    print(' ###                         Step3: Save to .vts file                        ### ')
    print(' ### ======================================================================= ### ')
    t0 = time.time()
    nr = n_r
    r_list = np.linspace(1, Rs, nr)
    rr, tt, pp = np.meshgrid(r_list, T_resample, P_resample, indexing='ij')
    br = np.zeros_like(rr)+0*1j
    bt = np.zeros_like(tt)+0*1j
    bp = np.zeros_like(pp)+0*1j
    tt = 90*deg-tt
    
    xx = rr*np.sin(tt)*np.cos(pp)
    yy = rr*np.sin(tt)*np.sin(pp)
    zz = rr*np.cos(tt)
    
    Bx,By,Bz = brtp2bxyz([Br,Bt,Bp],[rr,90*deg-tt,pp])
    vts_file = 'pfss'
    gridToVTK(vts_file, xx, yy, zz,
              pointData={
                  'B_xyz': (Bx,  By,  Bz),
                  'B_rtp': (Br, Bt, Bp),
                  'Br': Br,
                  'Bp': Bp,
                  'Bt': Bt
              }
              )
    print(f'Time Used: {(time.time()-t0)/60:8.3} min...')

if save_vtu and not skip_Brtp:
    print(' ### ======================================================================= ### ')
    print(' ###                         Step4: Save to .vtu file                        ### ')
    print(' ### ======================================================================= ### ')
    t0 = time.time()
    if not save_vts:
        nr = n_r
        r_list = np.linspace(1, Rs, nr)
        rr, tt, pp = np.meshgrid(r_list, T_resample, P_resample, indexing='ij')
        tt = 90*deg-tt
        
        xx = rr*np.sin(tt)*np.cos(pp)
        yy = rr*np.sin(tt)*np.sin(pp)
        zz = rr*np.cos(tt)
        
        Bx,By,Bz = brtp2bxyz([Br,Bt,Bp],[rr,90*deg-tt,pp])
    
    B_r=Br
    B_t=Bt
    B_p=Bp
    grid = vtk.vtkUnstructuredGrid()
    
    points = vtk.vtkPoints()
    
    # 创建一个vtk的FloatArray对象，用于存储物理量数据
    br_array = vtk.vtkFloatArray()
    bt_array = vtk.vtkFloatArray()
    bp_array = vtk.vtkFloatArray()
    bx_array = vtk.vtkFloatArray()
    by_array = vtk.vtkFloatArray()
    bz_array = vtk.vtkFloatArray()
    
    # 设置每个FloatArray的名称
    br_array.SetName("Br")
    bt_array.SetName("Bt")
    bp_array.SetName("Bp")
    bx_array.SetName("Bx")
    by_array.SetName("By")
    bz_array.SetName("Bz")
    
    B_array = vtk.vtkFloatArray()
    B_array.SetNumberOfComponents(3)
    B_array.SetName("Bvec")
    
    n1,n2,n3 = xx.shape
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                points.InsertNextPoint(xx[i,j,k], yy[i,j,k], zz[i,j,k])
                # 将你的物理量数据添加到FloatArray对象中
                br_array.InsertNextValue(B_r[i,j,k])
                bt_array.InsertNextValue(B_t[i,j,k])
                bp_array.InsertNextValue(B_p[i,j,k])
                bx_array.InsertNextValue(Bx[i,j,k])
                by_array.InsertNextValue(By[i,j,k])
                bz_array.InsertNextValue(Bz[i,j,k])
                B_array.InsertNextTuple([Bx[i,j,k], By[i,j,k], Bz[i,j,k]])
                # 添加六面体单元格到grid对象
                if i<n1-1 and j<n2-1:
                    hexahedron = vtk.vtkHexahedron()
                    hexahedron.GetPointIds().SetId(0, i*n2*n3 + j*n3 + k)
                    hexahedron.GetPointIds().SetId(1, (i+1)*n2*n3 + j*n3 + k)
                    hexahedron.GetPointIds().SetId(2, (i+1)*n2*n3 + (j+1)*n3 + k)
                    hexahedron.GetPointIds().SetId(3, i*n2*n3 + (j+1)*n3 + k)
                    hexahedron.GetPointIds().SetId(4, i*n2*n3 + j*n3 + (k+1)%n3)
                    hexahedron.GetPointIds().SetId(5, (i+1)*n2*n3 + j*n3 + (k+1)%n3)
                    hexahedron.GetPointIds().SetId(6, (i+1)*n2*n3 + (j+1)*n3 + (k+1)%n3)
                    hexahedron.GetPointIds().SetId(7, i*n2*n3 + (j+1)*n3 + (k+1)%n3)
                    grid.InsertNextCell(hexahedron.GetCellType(), hexahedron.GetPointIds())
    
    
    # 将points对象设置为grid的点集
    grid.SetPoints(points)
    
    # 将FloatArray对象添加到grid的单元数据中
    grid.GetPointData().AddArray(br_array)
    grid.GetPointData().AddArray(bt_array)
    grid.GetPointData().AddArray(bp_array)
    grid.GetPointData().AddArray(bx_array)
    grid.GetPointData().AddArray(by_array)
    grid.GetPointData().AddArray(bz_array)
    grid.GetPointData().AddArray(B_array)
     
    # 创建一个vtk的XMLUnstructuredGridWriter对象，用于写入VTK文件
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName("pfss.vtu")
    writer.SetInputData(grid)
    writer.Write()
    
print(f'Time Used: {(time.time()-t0)/60:8.3} min...')
