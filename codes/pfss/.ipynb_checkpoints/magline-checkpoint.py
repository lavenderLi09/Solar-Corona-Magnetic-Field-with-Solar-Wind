import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

from skimage import measure
from multiprocessing import Manager
from plotly.subplots import make_subplots
from concurrent.futures import ProcessPoolExecutor, as_completed
from .needs import *
from .funcs import matplotlib_to_plotly, trilinear_interpolation
from . import geometry

def rk45(rfun, x, dl, **kwargs):
    sig = kwargs.get('sig', 1)
    x0  = x
    k1  = rfun(x0            , **kwargs)
    # print(k1)
    k2  = rfun(x0+sig*k1*dl/2, **kwargs)
    k3  = rfun(x0+sig*k2*dl/2, **kwargs)
    k4  = rfun(x0+sig*k3*dl  , **kwargs)
    k   = (k1+2*k2+2*k3+k4)/6*sig
    ret = x0+k*dl
    ret[1] = ret[1] % np.pi
    ret[2] = ret[2] %(np.pi*2)
    return ret

def magline_stepper(instance, rtp, **kwargs):
    # print('rtp :' ,rtp)
    eps   = kwargs.get('eps', 1e-10)
    if np.any(np.isnan(rtp)):
        return np.full(3,np.nan)
    r,t,p = rtp
    Brtp  = instance.get_Brtp(rtp, **kwargs)
    Bn    = np.linalg.norm(Brtp)
    if Bn<eps:
        return np.full(3,np.nan)
    Bhat  = Brtp/np.linalg.norm(Brtp, axis=0)
    ret   = Bhat/np.array([1,r,r*np.sin(t)])
    return ret

def magline_solver(instance, rtps, **kwargs):
    # print(instance)
    dl   = kwargs.get('step_length', 0.003)
    Rl   = kwargs.get('Rl'         , 1.0  )
    Rs   = kwargs.get('Rs'         , 2.5  )
    Ns   = kwargs.get('max_steps'  , 10000)
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
            rtp0 = rk45(instance.magline_stepper, rtp0, dl, sig= 1, **kwargs)
            forward.append(rtp0)
        # backward integral
        rtp0 = irtp
        for i in range(Ns):
            if rtp0[0]<Rl or rtp0[0]>Rs or np.any(np.isnan(rtp0)):
                break
            rtp0 = rk45(instance.magline_stepper, rtp0, dl, sig=-1, **kwargs)
            backward.append(rtp0)
        imagline = np.array(backward[::-1]+forward)
        maglines.append(imagline)
    return maglines

def magline_task(instance, irtp, Rl, Rs, dl, Ns):
    return magline_solver(instance, irtp, Rl=Rl, Rs=Rs, step_length=dl, max_steps=Ns)

def parallel_magline_solver(instance, rtps, **kwargs):
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
            futures.append(executor.submit(magline_task, instance, irtp, Rl, Rs, dl, Ns))
        for future in futures:
            magline_res.extend(future.result())
    print(f'Time Used: {(time.time()-t0)/60:8.3f} min')
    return magline_res
        

def show_boundary(instance, **kwargs):
    vmin   = kwargs.get('vmin'  , None      )
    vmax   = kwargs.get('vmax'  , None      )
    cmap   = kwargs.get('cmap'  , 'coolwarm')
    show   = kwargs.get('show'  , True      )
    fsize  = kwargs.get('fsize' , 14        )
    width  = kwargs.get('width' , 800       )
    height = kwargs.get('height', 800       )
    frame  = kwargs.get('frame' , False     )
    cmap  = matplotlib_to_plotly(plt.get_cmap(cmap))

    tb,pb = np.mgrid[np.pi:0:200j, 0:2*np.pi:400j]
    rb    = 1.0
    xb    = rb*np.sin(tb)*np.cos(pb)
    yb    = rb*np.sin(tb)*np.sin(pb)
    zb    = rb*np.cos(tb)

    fig   = go.Figure(data=[go.Surface(x=xb,y=yb,z=zb, 
                                       surfacecolor=instance.Br_BB, 
                                       colorscale=cmap,
                                       cmin= vmin,
                                       cmax= vmax,
                                       colorbar=dict(
                                           title=dict(text='Br [G]',font=dict(size=fsize)),
                                           titleside='right',
                                           tickmode='auto',
                                       ),
                                       name='boundary'
                                      )])
    if not frame:
        fig = unframe(fig)
    if show:
        fig.update_layout(width =width,
                          height=height
                         )
        fig.show()
        return fig
    else:
        return fig

def show_maglines(instance, **kwargs):
    t0     = time.time()
    seeds  = kwargs.get('seeds'    , None    )
    nlines = kwargs.get('nlines'   , 100     )
    show   = kwargs.pop('show'     , True    )
    lw     = kwargs.get('lw'       , 5       )
    c      = kwargs.get('color'    , 'green' )
    R1     = kwargs.get('R1'       , 1.0     )
    R2     = kwargs.get('R2'       , 2.5     )
    width  = kwargs.get('width'    , 800     )
    height = kwargs.get('height'   , 800     )
    parl   = kwargs.get('parl'     , dict()  )
    frame  = kwargs.get('frame'    , True    )
    PY     = kwargs.get('python'   , 'python')
    if seeds is None:
        rs = np.random.uniform(R1, R2      , nlines)
        ts = np.random.uniform(0 , np.pi   , nlines)
        ps = np.random.uniform(0 , np.pi*2 , nlines)
        seeds = np.stack([rs,ts,ps], axis=1)
    instance.info['seeds']=seeds
    fig = kwargs.get('fig', instance.show_boundary(show=False, **kwargs))
    if not parl:
        maglines = instance.magline_solver(seeds,**kwargs)
    else:
        FM       = parl.get('fast_mode'  , True              )
        n_cores  = parl.get('n_cores'    , 80                )
        eps      = parl.get('eps'        , 1e-10             )
        dl       = parl.get('step_length', 0.003             )
        Ns       = parl.get('max_steps'  , 10000             )
        Rl       = parl.get('Rl'         , 1.0               )
        Rs       = parl.get('Rs'         , instance.Rtp      )
        SN       = parl.get('save_name'  , './maglines.npy'  )
        PN       = parl.get('pkl_name'   , instance.save_name)
        if not FM:
            maglines = parallel_magline_solver(instance, seeds, n_cores=n_cores, **kwargs)
        else:
            instance.info['maglines']=dict(seeds=seeds,
                                       n_cores=n_cores,
                                       eps=eps,
                                       step_length=dl,
                                       max_steps=Ns,
                                       Rl=Rl,
                                       Rs=Rs,
                                       save_name=SN,
                                      )
            instance.save(save_name=PN)
            if instance.__class__.__name__   == 'pfss_solver':
                T = 'ps'
            elif instance.__class__.__name__ == 'scs_solver':
                T = 'ss'
            elif instance.__class__.__name__ == 'off_solver':
                T = 'ofs'
            else:
                raise ValueError(f"Class: `{instance.__class__}` is not supported...")
            command  = PY+f" -u -m pfss.magline_script -i {PN} -t {T} > maglines.out 2>&1 &"
            tem_ret  = os.system(command)
            timeout  = 10
            time0    = time.time()
            while not os.path.exists(SN):
                # print(glob.glob('*.npy'))
                if time.time()-time0 > timeout:
                    raise TimeoutError(f"Wait timeout, file {SN} could not be generated within {timeout} seconds...")
                time.sleep(1)
            maglines = np.load(SN, allow_pickle=True)
            os.remove(SN)
            os.remove(PN)
            os.remove('maglines.out')
            ti = time.time()
            print(f'Stream Line Integrating Finished.  Time Used: {(ti-t0):7.3f} sec...')
                                       
    maglines_xyz = []
    for imagline in maglines:
        ri,ti,pi = imagline.T
        xi = ri*np.sin(ti)*np.cos(pi)
        yi = ri*np.sin(ti)*np.sin(pi)
        zi = ri*np.cos(ti)
        maglines_xyz.append(np.stack([xi,yi,zi], axis=1))
    if not instance.info.get('maglines', dict()):
        instance.info['maglines'] = dict(maglines_rtp=maglines,
                                         maglines_xyz=maglines_xyz)
    else:
        instance.info['maglines']['maglines_rtp']=maglines
        instance.info['maglines']['maglines_xyz']=maglines_xyz
    for imagline in maglines_xyz:
        xx,yy,zz = imagline.T
        fig.add_trace(go.Scatter3d(x=xx,
                                   y=yy,
                                   z=zz,
                                   mode='lines',
                                   showlegend=False,
                                   line=dict(color=c,width=lw)))
    if not frame:
        fig = unframe(fig)
    if show:
        fig.update_layout(width =width,
                          height=height
                         )
        fig.show()
        return fig
    else:
        return fig

def show_current_sheet(instance, **kwargs):
    rtp   = kwargs.get('rtp' , instance.get_rtp())
    Bfile = kwargs.pop('load_file', instance.scs_file)
    Brtp  = kwargs.get('Brtp', np.load(Bfile))
    fig   = kwargs.get('fig', go.Figure())
    show  = kwargs.get('show', True)
    cmap  = kwargs.get('cmap', 'jet')
    alpha = kwargs.get('alpha',0.9)
    width = kwargs.get('width',800)
    height= kwargs.get('height',800)
    frame = kwargs.get('frame', True)
    verts, faces, normals, values_on_faces = measure.marching_cubes(Brtp[0], 0)
    rv,tv,pv = trilinear_interpolation(rtp.transpose(1,2,3,0), verts).T
    xv   = rv*np.sin(tv)*np.cos(pv)
    yv   = rv*np.sin(tv)*np.sin(pv)
    zv   = rv*np.cos(tv)
    Jvec = geometry.rot(Brtp,rtp=rtp)
    Jmag = np.linalg.norm(Jvec,axis=0)
    Jv   = trilinear_interpolation(Jmag, verts)
    logJ = np.log10(Jv)
    vmin = kwargs.get('vmin', Jv.min())
    vmax = kwargs.get('vmax', Jv.max())
    vmin = np.log10(vmin)
    vmax = np.log10(vmax)
    ticks = kwargs.get('ticks', dict())
    fsize = ticks.get('fsize', 14)
    if ticks:
        ticks_value = ticks.get('ticks_value', None)
        ticks_label = ticks.get('ticks_label', None)
        if ticks_value is None:
            ticks_value = np.linspace(vmin, vmax, 5)
        else:
            ticks_value = np.log10(np.array(ticks_value))
        if ticks_label is None:
            ticks_label = [f"{il:.2e}" for il in np.power(10, ticks_value)]
    else:
        ticks_value = np.linspace(vmin, vmax, 5)
        ticks_label = [f"{il:.2e}" for il in np.power(10, ticks_value)]
    fig.add_trace(
        go.Mesh3d(
            x=xv,  # 顶点的 x 坐标
            y=yv,  # 顶点的 y 坐标
            z=zv,  # 顶点的 z 坐标
            i=faces[:,0],  # 三角形面片的第一个顶点索引
            j=faces[:,1],  # 三角形面片的第二个顶点索引
            k=faces[:,2],  # 三角形面片的第三个顶点索引
            opacity=alpha,  # 设置透明度
            intensity=logJ,
            colorscale=matplotlib_to_plotly(plt.get_cmap(cmap)),
            flatshading=True,  # 使用平面着色
            cmin=vmin,  # 设置颜色映射的最小值
            cmax=vmax,  # 设置颜色映射的最大值
            colorbar=dict(
                title=dict(text=r"|J|", font=dict(size=fsize)),  # 色条标题
                tickmode="array",  # 使用自定义刻度
                tickvals=ticks_value,  # 对数刻度
                ticktext=ticks_label,  # 对数刻度的标签
                titleside='right',
            ),
            name='current_sheet'
        )
    )
    if not frame:
        fig = unframe(fig)
    if show:
        fig.update_layout(width =width,
                          height=height
                         )
        fig.show()
    return fig

def unframe(fig, **kwargs):
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showgrid=False,       
                showline=False,       
                showticklabels=False, 
                ticks='',             
                showbackground=False, 
                title='',
            ),
            yaxis=dict(
                showgrid=False, 
                showline=False, 
                showticklabels=False, 
                ticks='', 
                showbackground=False,
                title='',
            ),
            zaxis=dict(
                showgrid=False, 
                showline=False, 
                showticklabels=False, 
                ticks='', 
                showbackground=False,
                title='',
            )
        )
    )
    return fig