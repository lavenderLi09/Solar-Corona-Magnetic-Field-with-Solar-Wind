from pfss.needs import *
from pfss.off_module import off_solver
from pfss.funcs import Associated_Legendre, DAssociated_Legendre

# ============================
parser = argparse.ArgumentParser(description='Computing PFSS model')
parser.add_argument('-i', type=str, help='Path to .pkl file for off_solver', required=False, default='off_solver.pkl')
parser.add_argument('-n', type=int, help='Devices Index'                   , required=False, default=0               )

t0        = time.time()
args      = parser.parse_args()
load_name = args.i
idx_D     = args.n
ofs       = off_solver.load(load_name=load_name)
lmax      = ofs.lmax
Alm,Blm   = ofs.Alm, ofs.Blm
HL,GL     = ofs.info['HL'], ofs.info['GL']
Nr,Nt,Np  = ofs.n_r,ofs.n_t,ofs.n_p
devices   = ofs.info['devices']
device    = devices[idx_D]
device    = torch.device(device)
mlist     = ofs.info['mlist']
mi,mf     = mlist[idx_D], mlist[idx_D+1]
rr,tt,pp  = ofs.get_rtp()
rls       = rr[:,0,0]
tls       = tt[0,:,0]
pls       = pp[0,0,:]
rr,tt,pp  = torch.from_numpy(np.stack([rr,tt,pp])).to(device)

br = torch.zeros_like(rr, dtype=torch.float64).to(device)
bt = torch.zeros_like(tt, dtype=torch.float64).to(device)
bp = torch.zeros_like(pp, dtype=torch.float64).to(device)
for m in range(mi, mf):
    for l in range(m, lmax+1):
        alm = Alm[l][m]
        blm = Blm[l][m]
        if l==m:
            P_l00 = Associated_Legendre(l,  m, torch.cos(tt))
            P_lp1 = Associated_Legendre(l+1,m, torch.cos(tt))
            cosmp = torch.cos(m*pp)
            sinmp = torch.sin(m*pp)
            Hl    = HL[l]
            Gl    = GL[l]
            Hl    = Hl[:,np.newaxis,np.newaxis].repeat(Nt,axis=1).repeat(Np,axis=2)
            Gl    = Gl[:,np.newaxis,np.newaxis].repeat(Nt,axis=1).repeat(Np,axis=2)
            Hl    = torch.from_numpy(Hl).to(device)
            Gl    = torch.from_numpy(Gl).to(device)
            Qlm   = P_l00
            DQlm  = DAssociated_Legendre(l,m,torch.cos(tt),P_l00=P_l00,P_lp1=P_lp1)
            Pm1   = sinmp
            Pm2   = cosmp
            Dpm1  = m*cosmp
            Dpm2  =-m*sinmp
            if l==0:
                br+=blm*Gl*Qlm*Pm2
            else:
                br+=Gl*Qlm*(alm*Pm1+blm*Pm2)
                bt+=-Hl*DQlm*torch.sin(tt)*(alm*Pm1+blm*Pm2)
                bp+=Hl/torch.sin(tt)*Qlm*(alm*Dpm1+blm*Dpm2)
            P_ln1 = P_l00
            P_l00 = P_lp1
        else:
            P_lp1 = Associated_Legendre(l+1, m, torch.cos(tt), pn2=P_ln1, pn1=P_l00)
            cosmp = torch.cos(m*pp)
            sinmp = torch.sin(m*pp)
            Hl    = HL[l]
            Gl    = GL[l]
            Hl    = Hl[:,np.newaxis,np.newaxis].repeat(Nt,axis=1).repeat(Np,axis=2)
            Gl    = Gl[:,np.newaxis,np.newaxis].repeat(Nt,axis=1).repeat(Np,axis=2)
            Hl    = torch.from_numpy(Hl).to(device)
            Gl    = torch.from_numpy(Gl).to(device)
            Qlm   = P_l00
            DQlm  = DAssociated_Legendre(l,m,torch.cos(tt),P_l00=P_l00,P_lp1=P_lp1)
            Pm1   = sinmp
            Pm2   = cosmp
            Dpm1  = m*cosmp
            Dpm2  =-m*sinmp
            br+=Gl*Qlm*(alm*Pm1+blm*Pm2)
            bt+=-Hl*DQlm*torch.sin(tt)*(alm*Pm1+blm*Pm2)
            bp+=Hl/torch.sin(tt)*Qlm*(alm*Dpm1+blm*Dpm2)
            P_ln1 = P_l00
            P_l00 = P_lp1
br   =  np.real(br.detach().cpu().numpy())
bt   =  np.real(bt.detach().cpu().numpy())
bp   =  np.real(bp.detach().cpu().numpy())
brtp =  np.stack([br,bt,bp], axis=0)
save_path = './OFF_temp_files/'
os.makedirs(save_path, exist_ok=True)
np.save(os.path.join(save_path,f'off_{mi:03d}_{mf:03d}.npy'), brtp)
ti = time.time()
print(f'mi: {mi:3d}, mf: {mf-1:3d}, wall_time:{(ti-t0)/60:8.3f} min', flush=True)