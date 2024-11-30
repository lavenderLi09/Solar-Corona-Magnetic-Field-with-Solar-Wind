'''
Code       : pfss_module.py
Date       : 2024.10.03
Contributer: G.Y.Chen(gychen@smail.nju.edu.cn)
Purpose    : Orgnize the functions about the process to build the PFSS model...

### --------------------------------- ###
Remark:
2024.10.03: Build the code (gychen@smail.nju.edu.cn)
2024.10.10: Add the subfunction, get_Brtp
2024.10.12: Add harmonics method in get_Brtp subroutine
2024.10.13: Add the magline visiualization function
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
import sys
import subprocess
import shutil
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

from .funcs import brtp2bxyz, trilinear_interpolation, Brtp_lm, Associated_Legendre
from .magline import rk45, magline_stepper, magline_solver, show_boundary, show_maglines

class pfss_solver():
    '''
    3D Potential Field Source Surface (PFSS) model solver
    
    Parameters:
        fits_file (str) -- the .fits file name for the syntopic magnetic map
        nr (int)        -- the grid solution in the r direction, default is 400
        nt (int)        -- the grid solution in the theta direction, default is 200
        np (int)        -- the grid solution in the phi direction, default is 400
        lmax (int)      -- the maximum l level for the Sphrical Harmonics Y_l^m, default is 80
        Rs (float)      -- Source Surface radius, deflaut is 2.5
    '''
    def __init__(self,
                 fits_file=None,
                 n_r  = 400,
                 n_t  = 200,
                 n_p  = 400,
                 lmax = 80,
                 Rs   = 2.5,
                 Br   = None
                ):
        self.fits_file  = fits_file
        self.hmi_map    = sunpy.map.Map(fits_file) if fits_file is not None else None
        self.fits_dims  = self.hmi_map.data.shape  if fits_file is not None else None
        self.n_r        = n_r
        self.n_p        = n_p
        self.n_t        = n_t
        self.lmax       = lmax
        self.Br         = self._resampling_Br() if fits_file is not None else Br
        self.Br_SS      = None # Br in Source surface
        self.Br_BB      = None # Br in Bottom boudanry
        self.Rs         = Rs
        self.Brtp       = None
        self.Alm        = None
        self.Blm        = None
        self.fig        = None
        self.info       = {}
        self.save_name  = 'pfss_solver.pkl'
        self.pfss_file  = 'Brtp_pfss.npy'
        self.Rtp        = Rs

    def __getstate__(self):
        state = self.__dict__.copy()
        # 移除无法序列化的属性
        del state['fig']
        del state['hmi_map']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 重新加载或初始化属性
        self.fig     = None
        self.hmi_map = sunpy.map.Map(self.fits_file)

    def _resampling_Br(self):
        '''
        '''
        deg          = np.pi/180
        nth,nph      = self.fits_dims
        sin_theta_ls = np.linspace(-1,1,nth)
        theta_ls     = np.arcsin(sin_theta_ls)
        phi_ls       = np.linspace(0,2*np.pi, nph)
        Th,Ph        = np.meshgrid(theta_ls, phi_ls, indexing='ij')
        Br           = self.hmi_map.data
        nt_resample  = self.n_t
        np_resample  = self.n_p
        dth          = np.pi/nt_resample
        dph          = np.pi/np_resample*2
        T_resample   = np.linspace(-90*deg, 90*deg, nt_resample+1)[:nt_resample]+0.5*dth
        P_resample   = np.linspace(0, 2*np.pi, np_resample+1)[:np_resample]+0.5*dph
        TT,PP        = np.meshgrid(T_resample, P_resample, indexing='ij')
        p_idx        = PP/(2*np.pi/(nph-1))
        t_idx        = (np.sin(TT)+1)/(2/(nth-1))
        p_size       = p_idx.shape
        t_size       = t_idx.shape
        p_idx        = p_idx.flatten()
        t_idx        = t_idx.flatten()
        p_idx0       = np.floor(p_idx).astype(int)
        p_idx1       = p_idx0+1
        t_idx0       = np.floor(t_idx).astype(int)
        t_idx1       = t_idx0+1
        Br_resample  = Br[(t_idx0,p_idx0)]*(p_idx1-p_idx)*(t_idx1-t_idx)/(p_idx1-p_idx0)/(t_idx1-t_idx0)+\
                       Br[(t_idx1,p_idx0)]*(t_idx-t_idx0)*(p_idx1-p_idx)/(t_idx1-t_idx0)/(p_idx1-p_idx0)+\
                       Br[(t_idx0,p_idx1)]*(t_idx1-t_idx)*(p_idx-p_idx0)/(t_idx1-t_idx0)/(p_idx1-p_idx0)+\
                       Br[(t_idx1,p_idx1)]*(t_idx-t_idx0)*(p_idx-p_idx0)/(t_idx1-t_idx0)/(p_idx1-p_idx0)
        Br_resample  = Br_resample.reshape(p_size)
        return Br_resample

    def save(self, **kwargs):
        self.save_name = kwargs.get('save_name', self.save_name)
        directory = os.path.dirname(self.save_name)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(self.save_name, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print(f"Instance saved to {self.save_name}")

    @classmethod
    def load(cls, **kwargs):
        load_name = kwargs.get('load_name','pfss_solver.pkl')
        with open(load_name, 'rb') as input:
            return pickle.load(input)

    def plot(self, data, **kwargs):
        vmin  = kwargs.get('vmin' , None    )
        vmax  = kwargs.get('vmax' , None    )
        grid  = kwargs.get('grid' , False   )
        PIL   = kwargs.get('PIL'  , False   )
        title = kwargs.get('title', r'$B_r$')
        Br    = data
        ax    = kwargs.get('ax', None)
        if ax is None:
            fig, ax = plt.subplots()
        im    = ax.imshow(Br, origin='lower', cmap='coolwarm', extent=(0,360,-90,90), aspect='equal')
        if vmin is not None and vmax is not None:
            im.set_clim(vmin, vmax)
        else:
            vmin = Br.min()
            vmax = Br.max()
            cval = max(-vmin,vmax)
            im.set_clim(-cval, cval)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label=r'$B_r$ [G]')
        if PIL:
            ax.contour(Br, levels=[0], colors='gray', extent=(0,360,-90,90))
        ax.set_title(title)
        ax.set_xlabel('Carriton Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xticks([90,180,270])
        ax.set_yticks([-60,-30,0,30,60]) 
        ax.set_xticklabels(['$90\degree$','$180\degree$','$270\degree$'])
        ax.set_yticklabels(['$-60\degree$', '$-30\degree$', '$0\degree$', '$30\degree$', '$60\degree$']) 
        if grid:
            ax.grid(color='w', alpha=0.3)
        return ax
        

    def plot_BC(self, **kwargs):
        title = kwargs.pop('title', r'HMI Bottom Boundary $B_r$')
        self.plot(self.Br, title=title, **kwargs)

    def plot_bottom_boundary(self, **kwargs):
        Br = self.Br_BB
        title = kwargs.pop('title', r'PFSS Bottom Boundary $B_r$')
        self.plot(Br, title=title, **kwargs)

    def plot_source_surface(self, **kwargs):
        Br    = self.Br_SS
        grid  = kwargs.pop('grid', True)
        PIL   = kwargs.pop('PIL' , True)
        title = kwargs.pop('title', r'Source Surface $B_r$')
        self.plot(Br,title=title, grid=grid,PIL=PIL, **kwargs)

    def get_pfss(self, **kwargs):
        '''
        To compute the PFSS solution of l order

        Parameters:
            lmax (int)       --  lmax for PFSS calculation, default is the initial value you set
            err  (float)     --  the minimum acceptable error in the calculation, default is 1e-200
            save_coef (bool) --  is save the coefficients Alm and Blm?  default is `False`
            n1_print (int)   --  the print interval in coefficient computing process
            n2_print (int)   --  the print interval in field building process
            n_cores (int)    --  the cpu cores number used to parallel computing
            device (str)     --  the device used to build the field. default is None, it means auto choice.
            Rs (float)       --  the radia of Source Surface
            fast_mode (bool) --  is use the advanced fast mode to build the field

        Return:
            Brtp (array)     -- array include 3 spherical component of vector magnetic field, [Br, Bt, Bp]
        '''
        F          = self.fits_file
        NR         = self.n_r
        NT         = self.n_t
        NP         = self.n_p
        L          = kwargs.get('lmax',self.lmax)
        err        = kwargs.get('err', 1e-200)
        S          = kwargs.get('save_coef', False)
        N1         = kwargs.get('n1_print', 1000)
        N2         = kwargs.get('n2_print', 1000)
        NC         = kwargs.get('n_cores', 10)
        D          = kwargs.get('device', None)
        RS         = kwargs.get('Rs', self.Rs)
        FM         = kwargs.get('fast_mode', True)
        LC         = kwargs.get('load_coef', False)
        SB         = kwargs.get('skip_Brtp', False)
        PF         = kwargs.get('pfss_file', self.pfss_file)
        if D is None:
            D      = 'cuda'  if torch.cuda.is_available() else 'cpu'
        command    = f"python -u -m pfss.pfss_script -f {F} -nr {NR} -nt {NT} -np {NP} -l {L} -rs {RS} --err {err} "+\
                     f"--n1_print {N1} --n2_print {N2} --n_cores {NC} --device {D}"
        command    = command if not S  else command + " --save_coef"
        command    = command if not FM else command + " --fast_mode"
        command    = command if not LC else command + " --load_coef"
        command    = command if not SB else command + " --skip_Brtp"
        command    = command if not SC else command + " --save_coef"
        ret        = os.system(command)
        Brtp       = np.load('./Brtp.npz')
        Br         = Brtp['Br']
        Bp         = Brtp['Bp']
        Bt         = Brtp['Bt']
        self.Br_SS = Br[-1]
        self.Br_BB = Br[0]
        Brtp       = np.stack([Br,Bp,Bt], axis=0)
        # os.remove('./Brtp.npz')
        np.save(self.pfss_file, Brtp)
        self.Brtp  = Brtp
        return Brtp

    def load_Brtp(self,**kwargs):
        load_file = kwargs.get('load_file', './Brtp.npz')
        Brtp      = np.load(load_file)
        Brtp      = np.stack([Brtp['Br'],Brtp['Bt'],Brtp['Bp']], axis=0)
        self.Brtp = Brtp
        return Brtp

    def get_rtp(self):
        dth      = np.pi/self.n_t
        dph      = np.pi/self.n_p
        deg      = np.pi/180
        r_list   = np.linspace( 1     , self.Rs, self.n_r)
        t_list   = np.linspace(-90*deg, 90*deg , self.n_t+1)[:-1]+dth/2
        p_list   = np.linspace( 0     , 2*np.pi, self.n_p+1)[:-1]+dph/2
        t_list   = np.pi/2-t_list
        rr,tt,pp = np.meshgrid(r_list,t_list,p_list, indexing='ij')
        return np.stack([rr,tt,pp])

    def save_vts(self, **kwargs):
        vts_name = kwargs.pop('vts_name', 'pfss')
        Brtp = kwargs.pop('Brtp', None)
        if Brtp is None:
            Brtp = np.load(self.pfss_file)
            # Brtp = np.load('./Brtp.npz')
            # Br   = Brtp['Br']
            # Bt   = Brtp['Bt']
            # Bp   = Brtp['Bp']
        else:
            Br,Bt,Bp = Brtp
        rr,tt,pp = self.get_rtp()
        Brtp2vts([Br,Bt,Bp], [rr,tt,pp], vts_name=vts_name, **kwargs)

    def save_vtu(self, **kwargs):
        vtu_name = kwargs.pop('vtu_name', 'pfss')
        Brtp     = kwargs.pop('Brtp', None)
        if Brtp is None:
            Brtp = np.load('./Brtp.npz')
            Br   = Brtp['Br']
            Bt   = Brtp['Bt']
            Bp   = Brtp['Bp']
        else:
            Br,Bt,Bp = Brtp
        rr,tt,pp = self.get_rtp()
        Brtp2vtu([Br,Bt,Bp], [rr,tt,pp], vtu_name=vtu_name, **kwargs)

    def multi_gpu_pfss(self, devices, **kwargs):
        t0 = time.time()
        F  = self.fits_file
        L  = kwargs.get('lmax'     , self.lmax)
        LC = kwargs.get('load_coef', True     )
        SS = kwargs.get('save_vts' , False    )
        SU = kwargs.get('save_vtu' , False    )
        N1 = kwargs.get('n1_print' , 100      )
        N2 = kwargs.get('n2_print' , 100      )
        FM = kwargs.get('fast_mode', False    )
        PY = kwargs.get('python'   , 'python ')
        SC = kwargs.get('save_coef', False    )
        PF = kwargs.get('pfss_file', self.pfss_file)
        BD = kwargs.get('bound'    , None     )
        R  = kwargs.get('rtp'      , None     )
        Ds = devices
        self.pfss_file=PF

        lm_list = [[il, im] for il in range(L + 1) for im in range(-il, il + 1)]
        total_tasks = len(lm_list)
        nD = len(Ds)
        chunks = total_tasks//nD
        res_tasks = total_tasks % nD
        chunk_list = [chunks]*nD
        for i in range(res_tasks):
            chunk_list[i]+=1
        if FM:
            lmax=L
            di = (L+1)*(L+2)/2/nD
            m_assigned = [0]
            icnt = 0
            for m in range(lmax+1):
                icnt+=lmax+1-m
                if icnt > di:
                    m_assigned.append(m+1)
                    icnt = icnt % di
            m_assigned.append(lmax+1)
            icnt=0
        idx_i = 0
        idx_f = 0
        commands = []
        os.makedirs('./Brtp_temp_files/', exist_ok=True)
        log_files = [os.path.join('./Brtp_temp_files',f'{i:04}.out') for i in range(nD)]
        part = PY+f" -u -m pfss.pfss_script -f {F} -l {L} --n1_print {N1} --n2_print {N2} "
        if BD is not None:
            np.save('temp_bound.npy', BD)
            part = part+' --bound temp_bound.npy '
        if R is not None:
            np.save('temp_rtp.npy', R)
            part = part+' --rtp temp_rtp.npy '
        option = " "
        option = option + '--save_vts '  if SS else option
        option = option + '--save_vtu '  if SU else option
        option = option + '--load_coef ' if LC else option
        option = option + '--fast_mode ' if FM else option
        option = option + ' --save_coef' if SC else option
        for i,lf in enumerate(log_files):
            if not FM:
                idx_i = idx_f
                idx_f = idx_f+chunk_list[i]
            else:
                idx_i = m_assigned[i  ]
                idx_f = m_assigned[i+1]
            D = Ds[i]
            command = part+f"--device {D} --start {idx_i} --end {idx_f}"+option+f" >{lf} 2>&1"
            commands.append(command)

        # print(commands)
        processes = []
        for command in commands:
            process = subprocess.Popen(command, shell=True)
            processes.append(process)
        
        for process in processes:
            process.wait()

        Brtp_files = sorted(glob.glob('./Brtp_temp_files/*.npz'))
        first = True
        for file in Brtp_files:
            Brtp = np.load(file)
            if first:
                Br = Brtp['Br']
                Bt = Brtp['Bt']
                Bp = Brtp['Bp']
                first = False
            else:
                Br += Brtp['Br']
                Bt += Brtp['Bt']
                Bp += Brtp['Bp']
        np.savez('./Brtp.npz',**dict(Br=Br,Bt=Bt,Bp=Bp))
        print(f'!!! Build the PFSS model successfully.  Total Time: {(time.time()-t0)/60:8.3} min !!!')
        out_files = sorted(glob.glob('./Brtp_temp_files/*.out'))
        with open('log.out', 'w') as outfile:
            for i,fname in enumerate(out_files):
                outfile.write(f"### {fname} Content :\n")
                with open(fname) as infile:
                    outfile.write(infile.read())
                    outfile.write('\n')
        shutil.rmtree('./Brtp_temp_files')
        self.Br_SS = Br[-1]
        self.Br_BB = Br[0]
        self.Brtp  = np.stack([Br,Bt,Bp], axis=0)
        np.save(PF, self.Brtp)
        return np.stack([Br,Bt,Bp], axis=0)

    def get_Brtp(self, rtp, **kwargs):
        method   = kwargs.get('method'   , 'interpolation')
        Bfile    = kwargs.get('load_file', './Brtp.npz'   )
        # print('Bfile: ', Bfile)
        rr,tt,pp = rtp
        if method=='interpolation':
            if self.Brtp is None:
                Brtp     = np.load(Bfile)
                Br       = Brtp['Br']
                Bp       = Brtp['Bp']
                Bt       = Brtp['Bt']
                Brtp     = np.stack([Br,Bt,Bp], axis=-1)
                self.Brtp = Brtp.transpose(3,0,1,2)
            else:
                Brtp = self.Brtp.transpose(1,2,3,0)
            ir       = (rr   - 1)/(self.Rs-1)*(self.n_r-1)
            it       = (np.pi-tt)/np.pi*(self.n_t-1)
            ip       = (pp   - 0)/2/np.pi*(self.n_p-1)
            if isinstance(rr, np.ndarray):
                size = rr.shape
                ir   = ir.flatten()
                it   = it.flatten()
                ip   = ip.flatten()
                idx  = np.stack([ir,it,ip], axis=1)
                ret  = trilinear_interpolation(Brtp, idx).T
                ret  = ret.reshape(3,*size)
            elif isinstance(rr, (int, float, complex)):
                idx  = [ir,it,ip]
                ret  = trilinear_interpolation(Brtp, idx)
            else:
                raise ValueError('params `rtp` should have be a numpy array or Scalar...')
        elif method=='harmonics':
            device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            device = torch.device(device)
            err    = kwargs.get('err', 1e-200)
            rr = torch.from_numpy(rr).to(device)
            tt = torch.from_numpy(tt).to(device)
            pp = torch.from_numpy(pp).to(device)
            Br = torch.zeros_like(rr, dtype=torch.complex128).to(device)
            Bt = torch.zeros_like(tt, dtype=torch.complex128).to(device)
            Bp = torch.zeros_like(pp, dtype=torch.complex128).to(device)
            Alm = kwargs.get('Alm', self.Alm)
            Blm = kwargs.get('Blm', self.Blm)
            if (Alm is None) or (Blm is None):
                with open('Alm.pkl', 'rb') as f:
                    Alm = pickle.load(f)
                with open('Blm.pkl', 'rb') as f:
                    Blm = pickle.load(f)
            lmax = max(Alm.keys())
            for m in range(0,lmax+1):
                for l in range(m,lmax+1):
                    if l==m:
                        P_l00 = Associated_Legendre(l,  m, torch.cos(tt))
                        P_lp1 = Associated_Legendre(l+1,m, torch.cos(tt))
                        br,bt,bp = Brtp_lm(l,m,rr,tt,pp,err=err,device=device,P_l00=P_l00,P_lp1=P_lp1,Alm=Alm,Blm=Blm)
                        Br+=br
                        Bt+=bt
                        Bp+=bp
                        if l!=0:
                            br,bt,bp = Brtp_lm(l,-m,rr,tt,pp,err=err,device=device,P_l00=P_l00,P_lp1=P_lp1,Alm=Alm,Blm=Blm)
                            Br+=br
                            Bt+=bt
                            Bp+=bp
                        P_ln1 = P_l00
                        P_l00 = P_lp1
                    else:
                        P_lp1 = Associated_Legendre(l+1, m, torch.cos(tt), pn2=P_ln1, pn1=P_l00)
                        br,bt,bp = Brtp_lm(l,m,rr,tt,pp,err=err,device=device,P_l00=P_l00,P_lp1=P_lp1,Alm=Alm,Blm=Blm)
                        Br+=br
                        Bt+=bt
                        Bp+=bp
                        if l!=0:
                            br,bt,bp = Brtp_lm(l,-m,rr,tt,pp,err=err,device=device,P_l00=P_l00,P_lp1=P_lp1,Alm=Alm,Blm=Blm)
                            Br+=br
                            Bt+=bt
                            Bp+=bp
                        P_ln1 = P_l00
                        P_l00 = P_lp1

            Br  = -np.real(Br.detach().cpu().numpy())
            Bt  = -np.real(Bt.detach().cpu().numpy())
            Bp  = -np.real(Bp.detach().cpu().numpy())
            ret =  np.stack([Br,Bt,Bp], axis=0)
        else:
            raise ValueError("params `method` should be either 'interpolation' or 'harmonics'...")
        return ret

    def magline_stepper(self, rtp, **kwargs):
        '''
        One Stepping solve the stream line function in sphrical coordinate system

        Parameters:
            rtp:  A sphrical coordinate point
            eps:  The minimum tolerant error, default is 1e-10
        '''
        return magline_stepper(self, rtp, **kwargs)

    def magline_solver(self, rtps, **kwargs):
        '''
        To return the magline according to the given magline seeds `rtps`

        Parameters:
            rtps:  The sphrical coordinate points
            Rl  :  The lower boundary radius, default is 1.0
            Rs  :  The upper boundary radius, default is 2.5
            Ns  :  Maximum integral steps

        Example:
            from pfss.pfss_module import pfss_solver
            ps       = pfss_solver(fits_file)
            Brtp     = ps.get_pfss(fast_mode=True)
            maglines = ps.magline_solver(user_magline_seeds)
        '''
        return magline_solver(self, rtps, **kwargs)

    def show_boundary(self, **kwargs):
        '''
        To show the magnetic field Br distributon on the Sun's surface

        Parameters:
            vmin   :  Seeds for stream line integrating
            vmax   :  If don't give a user's seeds, randomly choose `nlines` seeds
            cmap   :  Color map to use, matplotlib cmap name, default is 'coolwarm'
            show   :  To show the final figure
            fsize  :  Fontsize of the colorbar label, default is 14
            width  :  Plot height, default is 800
            height :  Plot height, default is 800

        Example:
            from pfss.pfss_module import pfss_solver
            ps       = pfss_solver(fits_file)
            Brtp     = ps.get_pfss(fast_mode=True)
            ps.show_boundary()
        '''
        return show_boundary(self, **kwargs)

    def show_maglines(self, **kwargs):
        '''
        To show the magnetic field lines of the PFSS model

        Parameters:
            seeds  :  Seeds for stream line integrating
            nlines :  If don't give a user's seeds, randomly choose `nlines` seeds
            Rl     :  The lower boundary radius, default is 1.0
            Rs     :  The upper boundary radius, default is 2.5
            show   :  To show the final figure, default is True
            lw     :  Line width to show the maglines, default is 5
            color  :  Colors of maglines, default is 'green'
            Ns     :  Maximum integral steps

        Example:
            from pfss.pfss_module import pfss_solver
            ps       = pfss_solver(fits_file)
            Brtp     = ps.get_pfss(fast_mode=True)
            ps.show_maglines()
        '''
        return show_maglines(self, **kwargs)
            
            

def Brtp2vts(Brtp,rtp, vts_name='pfss',**kwargs):
    print(' Start to save .vts file... ')
    t0 = time.time()
    Br,Bt,Bp = Brtp
    rr,tt,pp =  rtp
    if tt.min()<0:
        tt  = np.pi/2-tt # convert theta to range of [180°, 0°]
    
    xx = rr*np.sin(tt)*np.cos(pp)
    yy = rr*np.sin(tt)*np.sin(pp)
    zz = rr*np.cos(tt)
    
    Bx,By,Bz = brtp2bxyz([Br,Bt,Bp],[rr,tt,pp])
    save_J = kwargs.get('save_J', False)
    if not save_J:
        gridToVTK(vts_name, xx, yy, zz,
                  pointData={
                      'B_xyz': (Bx, By, Bz),
                      'B_rtp': (Br, Bt, Bp),
                      'Br': Br,
                      'Bp': Bp,
                      'Bt': Bt
                  }
                  )
    else:
        Jr,Jt,Jp = rot(Brtp,geometry='spherical',rtp=rtp)
        Jx,Jy,Jz = brtp2bxyz([Jr,Jt,Jp],[rr,tt,pp])
        gridToVTK(vts_name, xx, yy, zz,
                  pointData={
                      'B_xyz': (Bx, By, Bz),
                      'B_rtp': (Br, Bt, Bp),
                      'J_xyz': (Jx, Jy, Jz),
                      'J_rtp': (Jr, Jt, Jp),
                      'Br': Br,
                      'Bp': Bp,
                      'Bt': Bt
                  }
                  )
    print(f'Time Used: {(time.time()-t0)/60:8.3} min...')

def Brtp2vtu(Brtp,rtp,vtu_name='pfss', **kwargs):
    print(' Start to save .vtu file... ')
    save_J = kwargs.get('save_J', False)
    t0 = time.time()
    B_r,B_t,B_p = Brtp
    rr, tt, pp  = rtp
    if save_J:
        Jr,Jt,Jp = rot(Brtp,geometry='spherical',rtp=rtp)
        Jx,Jy,Jz = brtp2bxyz([Jr,Jt,Jp],[rr,tt,pp])
    if tt.min()<0:
        tt  = np.pi/2-tt # convert theta to range of [180°, 0°]  
    xx = rr*np.sin(tt)*np.cos(pp)
    yy = rr*np.sin(tt)*np.sin(pp)
    zz = rr*np.cos(tt)
    Bx,By,Bz = brtp2bxyz([B_r,B_t,B_p],[rr,tt,pp])

    grid = vtk.vtkUnstructuredGrid()    
    points = vtk.vtkPoints()
    br_array = vtk.vtkFloatArray()
    bt_array = vtk.vtkFloatArray()
    bp_array = vtk.vtkFloatArray()
    bx_array = vtk.vtkFloatArray()
    by_array = vtk.vtkFloatArray()
    bz_array = vtk.vtkFloatArray()
    br_array.SetName("Br")
    bt_array.SetName("Bt")
    bp_array.SetName("Bp")
    bx_array.SetName("Bx")
    by_array.SetName("By")
    bz_array.SetName("Bz")
    B_array = vtk.vtkFloatArray()
    B_array.SetNumberOfComponents(3)
    B_array.SetName("Bvec")
    if save_J:
        Jrtp_array = vtk.vtkFloatArray()
        Jrtp_array.SetNumberOfComponents(3)
        Jrtp_array.SetName('Jrtp')
        Jxyz_array = vtk.vtkFloatArray()
        Jxyz_array.SetNumberOfComponents(3)
        Jxyz_array.SetName('Jxyz')
    
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
                if save_J:
                    Jxyz_array.InsertNextTuple([Jx[i,j,k],Jy[i,j,k],Jz[i,j,k]])
                    Jrtp_array.InsertNextTuple([Jr[i,j,k],Jt[i,j,k],Jp[i,j,k]])
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
    
    
    grid.SetPoints(points)
    
    grid.GetPointData().AddArray(br_array)
    grid.GetPointData().AddArray(bt_array)
    grid.GetPointData().AddArray(bp_array)
    grid.GetPointData().AddArray(bx_array)
    grid.GetPointData().AddArray(by_array)
    grid.GetPointData().AddArray(bz_array)
    grid.GetPointData().AddArray(B_array)
     
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(vtu_name+".vtu")
    writer.SetInputData(grid)
    writer.Write()
    print(f'Time Used: {(time.time()-t0)/60:8.3} min...')