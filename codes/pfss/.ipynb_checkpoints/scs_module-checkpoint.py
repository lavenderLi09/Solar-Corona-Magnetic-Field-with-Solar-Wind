'''
Code       : scs_module.py
Date       : 2024.10.15
Contributer: H.Y.Li (liyuhua0909@126.com), G.Y.Chen (gychen@smail.nju.edu.cn)
Purpose    : Extending the PFSS model to a larger scale...

### --------------------------------- ###
Remark:
2024.10.15: Build the code
'''

from .needs import *

from .pfss_module import pfss_solver
from .funcs import brtp2bxyz, trilinear_interpolation, Brtp_lm, Associated_Legendre
from .magline import rk45, magline_stepper, magline_solver, show_boundary, show_maglines,parallel_magline_solver,show_current_sheet

# ================

def reorientation(Br_cp,Bt_cp,Bp_cp):
    mask_negative = Br_cp < 0
    Br_cp[mask_negative] = -Br_cp[mask_negative]
    Bt_cp[mask_negative] = -Bt_cp[mask_negative]
    Bp_cp[mask_negative] = -Bp_cp[mask_negative]
    return Br_cp, Bt_cp, Bp_cp

def Pnm(n, m, theta, **kwargs):
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    is_array = isinstance(theta, np.ndarray)
    if is_array:
        theta = torch.from_numpy(theta).to(device)
    delta = 0 if m!=0 else 1
    ret   = np.sqrt((2-delta)*float(np.math.factorial(n-m))/float(np.math.factorial(n+m)))
    ret   = ret*Associated_Legendre(n,m, torch.cos(theta), **kwargs)
    if is_array:
        return ret.detach().cpu().numpy()
    else:
        return ret

def DPnm(n, m, theta, **kwargs):
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    is_array = isinstance(theta, np.ndarray)
    if is_array:
        theta = torch.from_numpy(theta). to(device)
    P_lp0_m = kwargs.get('P_lp0_m', Associated_Legendre(n  ,m,torch.cos(theta),**kwargs))
    P_lp1_m = kwargs.get('P_lp1_m', Associated_Legendre(n+1,m,torch.cos(theta),**kwargs))
    dL_dth = 1/torch.sin(theta)*(-(n+1)*torch.cos(theta)*P_lp0_m+(n-m+1)*P_lp1_m)
    delta = 0 if m!=0 else 1
    ret   = np.sqrt((2-delta)*float(np.math.factorial(n-m))/float(np.math.factorial(n+m)))
    ret   = ret*dL_dth
    if is_array:
        return ret.detach().cpu().numpy()
    else:
        return ret

def alpha_beta(n,m,tt,pp,**kwargs):
    device    = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device    = torch.device(device)
    is_array  = isinstance(tt, np.ndarray) or isinstance(pp, np.ndarray)
    if is_array:
        tt    = torch.from_numpy(tt).to(device)
        pp    = torch.from_numpy(pp).to(device)
        
    P         =  Pnm(n,m,tt,**kwargs)
    dP_dth    = DPnm(n,m,tt,**kwargs)
    
    alpha_1nm = (n+1)*P*torch.cos(m*pp)
    alpha_2nm = -dP_dth*torch.cos(m*pp)
    alpha_3nm = m/torch.sin(tt)*P*torch.sin(m*pp)

    beta_1nm  = (n+1)*P*torch.sin(m*pp)
    beta_2nm  = -dP_dth*torch.sin(m*pp)
    beta_3nm  = m/torch.sin(tt)*P*torch.cos(m*pp)

    alpha     = torch.stack([alpha_1nm,alpha_2nm,alpha_3nm], dim=0)
    beta      = torch.stack([beta_1nm ,beta_2nm ,beta_3nm ], dim=0)
    if is_array:
        alpha = alpha.detach().cpu().numpy()
        beta  =  beta.detach().cpu().numpy()
    return alpha, beta

def get_alpha_beta_mat(th_list, ph_list, lmax=80, **kwargs):
    TT,PP = np.meshgrid(th_list, ph_list, indexing='ij')
    th,ph = TT.flatten(), PP.flatten()
    lm_list = [[il,im] for il in range(lmax+1) for im in range(il+1)]
    ret = []
    for l,m in lm_list:
        [alpha1,alpha2,alpha3],[beta1,beta2,beta3] = alpha_beta(l,m,th,ph, **kwargs)
        ret.append(np.hstack([alpha1,alpha2,alpha3]))
    lm_list = [[il,im] for il in range(1, lmax+1) for im in range(1,il+1)]
    for l,m in lm_list:
        [alpha1,alpha2,alpha3],[beta1,beta2,beta3] = alpha_beta(l,m,th,ph, **kwargs)
        ret.append(np.hstack([beta1,beta2,beta3]))
    ret = np.stack(ret, axis=0)
    return ret

def build_SCS_Brtp(rr,tt,pp,glm,hlm,lmax=10, **kwargs):
    device    = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device    = torch.device(device)
    Rcp       = kwargs.get('Rcp', 2.49)
    is_array = isinstance(rr, np.ndarray)
    if is_array:
        rr = torch.from_numpy(rr).to(device)
        pp = torch.from_numpy(pp).to(device)
        tt = torch.from_numpy(tt).to(device)
    br = torch.zeros_like(rr)
    bt = torch.zeros_like(tt)
    bp = torch.zeros_like(pp)
    for l in range(lmax+1):
        for m in range(l+1):
            plm  =  Pnm(l,m,tt)
            dplm = DPnm(l,m,tt)
            br+=(l+1)*(Rcp/rr)**(l+2)*plm*(glm[l][m]*torch.cos(m*pp)+hlm[l][m]*torch.sin(m*pp))
            bt+=-(Rcp/rr)**(l+2)*dplm*(glm[l][m]*torch.cos(m*pp)+hlm[l][m]*torch.sin(m*pp))
            bp+=(Rcp/rr)**(l+2)*plm*m/torch.sin(tt)*(glm[l][m]*torch.sin(m*pp)-hlm[l][m]*torch.cos(m*pp))
    Brtp = torch.stack([br,bt,bp],dim=0)
    if is_array:
        Brtp = Brtp.detach().cpu().numpy()
    return Brtp

# ================

class scs_solver(pfss_solver):
    def __init__(self,
                 fits_file,
                 n_r      = 400,
                 n_t      = 200,
                 n_p      = 400,
                 lmax     = 80,
                 Rs       = 2.5,
                 Rcp      = 2.4,
                 Rtp      = 10.,
                 lmax_scs = 10,
                 Nrtp_scs = [200,200,400]
                ):
        super().__init__(fits_file,n_r,n_t,n_p,lmax,Rs)
        self.Rcp       = Rcp
        self.Rtp       = Rtp
        self.lmax_scs  = lmax_scs
        self.glm       = None
        self.hlm       = None
        self.Nrtp_scs  = Nrtp_scs
        self.mask      = None
        self.scs_file  = './Brtp_scs.npy'
        self.save_name = 'scs_solver.pkl'
        self._initialization()

    def _initialization(self):
        lmax  = self.lmax_scs
        Nt,Np = self.Nrtp_scs[1:]
        Rcp   = self.Rcp
        dth   = np.pi/Nt
        dph   = np.pi/Np*2
        t_list   = np.linspace(0,  np.pi,Nt+1)[:-1]+0.5*dth
        p_list   = np.linspace(0,2*np.pi,Np+1)[:-1]+0.5*dph
        t_list   = t_list[::-1]
        Tcp,Pcp  = np.meshgrid(t_list,p_list,indexing='ij')
        rr,tt,pp = np.meshgrid(np.array([Rcp]), t_list, p_list, indexing='ij')
        rtp_cp   = np.stack([rr,tt,pp], axis=0)
        Brtp_cp  = super().get_Brtp(rtp_cp)
        Br_cp,Bt_cp,Bp_cp = Brtp_cp[:,0,:,:]
        self.mask= Br_cp<0
        Br_cp,Bt_cp,Bp_cp = reorientation(Br_cp, Bt_cp, Bp_cp)
        alpha_beta_mat = get_alpha_beta_mat(t_list, p_list, lmax=10)
        AB_mat = np.matmul(alpha_beta_mat, alpha_beta_mat.T)
        B_hat  = np.hstack([Br_cp.flatten(),Bt_cp.flatten(),Bp_cp.flatten()])
        GH_hat = np.matmul(np.linalg.inv(AB_mat),np.matmul(alpha_beta_mat,B_hat))
        glm = {il: {} for il in range(lmax + 1)}
        hlm = {il: {} for il in range(lmax + 1)}
        G   = GH_hat[:(lmax+1)*(lmax+2)//2]
        H   = GH_hat[(lmax+1)*(lmax+2)//2:]
        for l in range(lmax+1):
            for m in range(l+1):
                glm[l][m] = G[(l+1)*l//2+m]
        for l in range(lmax+1):
            hlm[l][0]=0
            for m in range(1,lmax+1):
                hlm[l][m] = H[l*(l-1)//2+m-1]
        self.glm = glm
        self.hlm = hlm

    def _inherited_from_pfss(self, ps):
        for key, value in ps.__dict__.items():
            if key in self.__dict__:
                self.__dict__[key] = value

    def get_rtp(self,**kwargs):
        Nrtp = kwargs.get('Nrtp', self.Nrtp_scs)
        Nr,Nt,Np = Nrtp
        dth = np.pi/Nt
        dph = np.pi/Np*2
        t_list = np.linspace(np.pi,0,Nt+1)[1:]+0.5*dth
        p_list = np.linspace(0,2*np.pi,Np+1)[:-1]+0.5*dph
        r_list = np.linspace(self.Rcp,self.Rtp,Nr)
        rr,tt,pp = np.meshgrid(r_list,t_list,p_list,indexing='ij')
        return np.stack([rr,tt,pp])

    def get_scs(self, **kwargs):
        print('Start to build the SCS field...')
        t0        = time.time()
        fname     = kwargs.get('fname', self.scs_file)
        lmax      = kwargs.pop('lmax', self.lmax_scs)
        glm       = self.glm
        hlm       = self.hlm
        rr,tt,pp  = self.get_rtp()
        Nr,Nt,Np  = self.Nrtp_scs
        Br,Bt,Bp  = build_SCS_Brtp(rr,tt,pp,glm,hlm,lmax=lmax,**kwargs)
        mask      = self.mask[np.newaxis,:,:].repeat(Nr,axis=0)
        ret       = np.stack([Br,Bt,Bp])
        ret[:,mask] = -ret[:,mask]
        self.scs_file = fname
        np.save(fname, ret)
        print(f'Finishing calculation takes {(time.time()-t0)/60:8.3f} min...')
        return ret

    def plot_cusp(self, **kwargs):
        Brtp = np.load(self.scs_file)
        Br   = Brtp[0][0]
        title = kwargs.pop('title',rf'Cusp Surface $B_r$ at {self.Rcp:.2f} $R_\odot$')
        self.plot(Br,title=title,**kwargs)

    def save_vts(self, **kwargs):
        vts_name = kwargs.pop('vts_name', 'csc')
        Brtp     = kwargs.pop('Brtp', np.load(self.scs_file))
        super().save_vts(vts_name=vts_name, Brtp=Brtp, **kwargs)

    def save_vtu(self, **kwargs):
        vtu_name = kwargs.pop('vtu_name', 'csc')
        Brtp     = kwargs.pop('Brtp', np.load(self.scs_file))
        super().save_vtu(vtu_name=vtu_name, Brtp=Brtp,**kwargs)

    def load_Brtp(self,**kwargs):
        Brtp = np.load(self.scs_file)
        return Brtp

    def get_Brtp(self, rtp, **kwargs):
        rtp      = np.stack(rtp)
        method   = kwargs.pop('method', 'interpolation')
        Bfile    = kwargs.pop('load_file', self.scs_file)
        Brtp     = np.load(Bfile)
        pfss     = np.load(kwargs.get('pfss_file', self.pfss_file))
        r,t,p    = rtp
        Nr,Nt,Np = self.Nrtp_scs
        if method!='interpolation':
            warnings.warn("`method` in SCS module supports only 'interpolation'.", UserWarning)
            method='interpolation'
        if isinstance(r,np.ndarray):
            region_pfss = r< self.Rcp
            region_scs  = r>=self.Rcp
            rtp_lower   = rtp[:,region_pfss]
            rtp_upper   = rtp[:,region_scs]
            rr,tt,pp    = rtp_lower
            Nr,Nt,Np    = pfss.shape[1:]
            ir          = (rr-1)/(self.Rs-1)*(Nr-1)
            it          = (np.pi-tt)/np.pi*(Nt-1)
            ip          = (pp-0)/2/np.pi*(Np-1)
            size        = rr.shape
            ir          = ir.flatten()
            it          = it.flatten()
            ip          = ip.flatten()
            idx         = np.stack([ir,it,ip], axis=1)
            ret_lower   = trilinear_interpolation(pfss.transpose(1,2,3,0), idx).T
            ret_lower   = ret_lower.reshape(3,*size)
            Nr,Nt,Np    = self.Nrtp_scs
            rr,tt,pp    = rtp_upper
            ir          = (rr-self.Rcp)/(self.Rtp-self.Rcp)*(Nr-1)
            it          = (np.pi-tt)/np.pi*(Nt-1)
            ip          = (pp-0)/2/np.pi*(Np-1)
            size        = rr.shape
            ir          = ir.flatten()
            it          = it.flatten()
            ip          = ip.flatten()
            idx         = np.stack([ir,it,ip], axis=1)
            ret_upper   = trilinear_interpolation(Brtp.transpose(1,2,3,0), idx).T
            ret_upper   = ret_upper.reshape(3, *size)
            ret         = np.zeros_like(rtp)
            ret[:,region_pfss] = ret_lower
            ret[:,region_scs]  = ret_upper
        elif isinstance(r, (int, float, complex)):
            if r<self.Rcp:
                ret = super().get_Brtp(rtp)
            else:
                ir  = (r-self.Rcp)/(self.Rtp-self.Rcp)*(Nr-1)
                it  = (np.pi-t)/np.pi*(Nt-1)
                ip  = (p-0)/2/np.pi*(Np-1)
                idx = [ir,it,ip]
                ret = trilinear_interpolation(Brtp.transpose(1,2,3,0), idx)
        else:
            raise ValueError('params `rtp` should have be a numpy array or Scalar...')
        return ret

    def show_maglines(self,**kwargs):
        Rs = kwargs.pop('Rs', self.Rtp)
        super().show_maglines(Rs=Rs, **kwargs)

    def parallel_magline_solver(self, rtps,**kwargs):
        return parallel_magline_solver(self, rtps, **kwargs)

    def show_current_sheet(self, **kwargs):
        return show_current_sheet(self, **kwargs)