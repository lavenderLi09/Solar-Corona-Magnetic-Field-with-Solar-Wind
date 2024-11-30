from .needs import *
from .pfss_module import pfss_solver
from .scs_module  import scs_solver
from .off_module  import off_solver
from .funcs import trilinear_interpolation

# =========================================
def rk45(rfun, x, dl, **kwargs):
    sig = kwargs.get('sig', 1)
    x0  = x
    k1  = rfun(x0            , **kwargs)
    k2  = rfun(x0+sig*k1*dl/2, **kwargs)
    k3  = rfun(x0+sig*k2*dl/2, **kwargs)
    k4  = rfun(x0+sig*k3*dl  , **kwargs)
    k   = (k1+2*k2+2*k3+k4)/6*sig
    ret = x0+k*dl
    ret[1] = ret[1] % np.pi
    ret[2] = ret[2] %(np.pi*2)
    return ret

def get_Brtp(rtp, **kwargs):
    r,t,p = rtp
    if load_type=='ss':
        Rcp   = kwargs.get('Rcp', instance.Rcp)
        if r<Rcp:
            ir   = (r   - 1)/(instance.Rs-1)*(instance.n_r-1)
            it   = (np.pi-t)/np.pi*(instance.n_t-1)
            ip   = (p   - 0)/2/np.pi*(instance.n_p-1)
            idx  = [ir,it,ip]
            ret  = trilinear_interpolation(pfss.transpose(1,2,3,0), idx)
        if r>=Rcp:
            Nr,Nt,Np = instance.Nrtp_scs
            ir   = (r-instance.Rcp)/(instance.Rtp-instance.Rcp)*(Nr-1)
            it   = (np.pi-t)/np.pi*(Nt-1)
            ip   = (p-0)/2/np.pi*(Np-1)
            idx  = [ir,it,ip]
            ret  = trilinear_interpolation(scs.transpose(1,2,3,0), idx)
    elif load_type=='ps':
        ir   = (r   - 1)/(instance.Rs-1)*(instance.n_r-1)
        it   = (np.pi-t)/np.pi*(instance.n_t-1)
        ip   = (p   - 0)/2/np.pi*(instance.n_p-1)
        idx  = [ir,it,ip]
        ret  = trilinear_interpolation(pfss.transpose(1,2,3,0), idx)
    elif load_type=='ofs':
        ir   = (r   - 1)/(instance.Rs-1)*(instance.n_r-1)
        it   = (np.pi-t)/np.pi*(instance.n_t-1)
        ip   = (p   - 0)/2/np.pi*(instance.n_p-1)
        idx  = [ir,it,ip]
        ret  = trilinear_interpolation(off.transpose(1,2,3,0), idx)
    else:
        raise ValueError("-t option can only support to be 'ss', 'ps' or 'ofs'...")
    return ret

def magline_stepper(rtp, **kwargs):
    # print('rtp :' ,rtp)
    eps   = kwargs.get('eps', 1e-10)
    if np.any(np.isnan(rtp)):
        return np.full(3,np.nan)
    r,t,p = rtp
    Brtp  = get_Brtp(rtp)
    Bn    = np.linalg.norm(Brtp)
    if Bn<eps:
        return np.full(3,np.nan)
    Bhat  = Brtp/np.linalg.norm(Brtp, axis=0)
    ret   = Bhat/np.array([1,r,r*np.sin(t)])
    return ret

def magline_solver(rtps, ntasks, dl=0.05, Rl=1.0, Rs=2.5, Ns=10000, lock=None,progress=None, t0=time.time()):
    rtps = np.array(rtps)
    maglines = []
    if rtps.ndim==1:
        rtps = rtps[None,:]
    for irtp in rtps:
        rtp0 = irtp
        forward  = [rtp0]
        backward = []
        # forward integral
        for i in range(Ns):
            if rtp0[0]<Rl or rtp0[0]>Rs or np.any(np.isnan(rtp0)):
                break
            rtp0 = rk45(magline_stepper, rtp0, dl, sig= 1)
            forward.append(rtp0)
        # backward integral
        rtp0 = irtp
        for i in range(Ns):
            if rtp0[0]<Rl or rtp0[0]>Rs or np.any(np.isnan(rtp0)):
                break
            rtp0 = rk45(magline_stepper, rtp0, dl, sig=-1)
            backward.append(rtp0)
        imagline = np.array(backward[::-1]+forward)
        maglines.append(imagline)
        with lock:
            ti = time.time()
            progress[0]+=1
            print(f"Tasks:{progress[0]:4d}/{ntasks:4d}, Line_point_nums: {len(imagline):5d}, Wall_time: {(ti-t0)/60:7.3f} min",
                  flush=True)
    return maglines


def parallel_magline_solver(rtps, **kwargs):
    t0 = time.time()
    dl   = kwargs.get('step_length', 0.05 )
    Rl   = kwargs.get('Rl'         , 1.0  )
    Rs   = kwargs.get('Rs'         , 10.0 )
    Ns   = kwargs.get('max_steps'  , 10000)
    total_tasks = len(rtps)
    n_cores     = min(kwargs.get('n_cores', 50), multiprocessing.cpu_count())
    chunks      = np.ceil(total_tasks/n_cores).astype(int)
    chunks_ls   = [chunks]*(total_tasks%n_cores)+[chunks-1]*(n_cores-total_tasks%n_cores)
    magline_res = []
    print('### =========== Parallel computing ============ ###')
    print(f'#       Available CPU cores: {multiprocessing.cpu_count():3d}                  #')
    print(f'#            Used CPU cores: {n_cores:3d}                  #')
    print('### =========================================== ###')

    manager     = Manager()
    progress    = manager.list([0])
    lock        = manager.Lock()

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = []
        for i in range(n_cores):
            idx_i = np.sum(chunks_ls[:i]  ).astype(int)
            idx_f = np.sum(chunks_ls[:i+1]).astype(int)
            irtp  = rtps[idx_i:idx_f]
            futures.append(executor.submit(magline_solver, irtp, total_tasks, dl, Rl, Rs, Ns, lock, progress, t0))
        for future in futures:
            magline_res.extend(future.result())
    print(f'Time Used: {(time.time()-t0)/60:8.3f} min')
    return magline_res

# =========================================

parser = argparse.ArgumentParser(description='Integrating magnetic field line')
parser.add_argument('-i' , type=str , help='Path to .pkl file for magnetic field solver', required=True               )
parser.add_argument('-t' , type=str , help='`ss` or `ps`, pfss or scs model'            , required=False, default='ss')

args      = parser.parse_args()
load_file = args.i
load_type = args.t

t0 = time.time()

if load_type=='ss':
    instance = scs_solver.load(load_name=load_file)
    pfss     = np.load(instance.pfss_file)
    scs      = np.load(instance.scs_file)
    pfss_rtp = pfss_solver.get_rtp(instance)
    scs_rtp  = instance.get_rtp()
    Rcp      = instance.Rcp
elif load_type=='ps':
    instance = pfss_solver.load(load_name=load_file)
    pfss     = np.load(instance.pfss_file)
    pfss_rtp = pfss_solver.get_rtp(instance)
elif load_type=='ofs':
    instance = off_solver.load(load_name=load_file)
    off      = np.load(instance.off_file)
    off_rtp  = instance.get_rtp()
else:
    raise ValueError("-t option can only support to be 'ss' or 'ps'...")

mag_info = instance.info['maglines']
seeds    = mag_info['seeds']
n_cores  = mag_info.get('n_cores', 50)
eps      = mag_info.get('eps', 1e-10)
dl       = mag_info.get('step_length', 0.05)
Ns       = mag_info.get('max_steps', 10000)
Rl       = mag_info.get('Rl',1.0 )
Rs       = mag_info.get('Rs',instance.Rtp)
magfile  = mag_info.get('save_name', 'maglines.npy')

maglines = parallel_magline_solver(seeds, 
                                   step_length=dl,
                                   max_steps=Ns,
                                   Rl=Rl,
                                   Rs=Rs,
                                   n_cores=n_cores
                                  )

maglines = np.array(maglines, dtype=object)
np.save(magfile, maglines)

ti = time.time()
print(f'Stream Line Integrating Finished.  Time Used: {(ti-t0):7.3f} sec...')