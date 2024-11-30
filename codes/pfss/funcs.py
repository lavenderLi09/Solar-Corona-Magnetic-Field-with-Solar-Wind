import math
import torch
import numpy as np

from scipy.special import sph_harm, lpmv
from pyevtk.hl import gridToVTK
from .geometry import rtp2xyz

def DSpherical_Harmonics(l,m,tt,pp,dim=0,**kwargs):
    is_array = False
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    if isinstance(tt, np.ndarray) or isinstance(pp, np.ndarray):
        tt = torch.from_numpy(tt).to(device)
        pp = torch.from_numpy(pp).to(device)
        is_array = True
    elif isinstance(tt, (int, float, complex)):
        tt = torch.tensor(tt).to(device)
        pp = torch.tensor(pp  ).to(device)
    Y_lm = kwargs.get('Y_lm', Spherical_Harmonics(l,m,tt,pp))
    if dim==1:
        ret = Y_lm*1j*m
    elif dim==0:
        Y_lp1_m = kwargs.get('Y_lp1_m', Spherical_Harmonics(l+1,m,tt,pp))
        ret = 1/torch.sin(tt)*(-(l+1)*Y_lm*torch.cos(tt)+(l-m+1)*Y_lp1_m*np.sqrt((2*l+1)*(l+m+1)/(2*l+3)/(l-m+1)))
    else:
        raise ValueError("`dim` can only be equal to 0 or 1, which mean differential in theta or phi")
    if is_array:
        return ret.detach().cpu().numpy()
    else:
        return ret

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
    # print(x)
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    is_array = kwargs.pop('is_array',None)
    if is_array is None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device)
            is_array = True
        else:
            is_array = False
    # print('first:', is_array)
    pn1    = kwargs.get('pn1', None)
    pn2    = kwargs.get('pn2', None)
    if pn1 is not None and pn2 is not None:
        plm = ((2*l-1)*x*pn1-(l+m-1)*pn2)/(l-m)
        return plm
    if np.abs(m)>l or torch.any(torch.abs(x)>1):
        raise ValueError("`m` cannot larget than `l`, or input |`x`| need to less than 1.")
    if m<0:
        m=-m
        plm = (-1)**m*np.math.factorial(l-m)/np.math.factorial(l+m)*Associated_Legendre(l,m,x,is_array=False,**kwargs)
    elif l==m:
        if l<=80:
            plm = (-1)**l*float(double_factorial(2*l-1))*(1-x**2)**(l/2)
        else:
            # print('test OK')
            plm = -(2*l-1)*torch.sqrt(1-x**2)*Associated_Legendre(l-1,l-1,x,is_array=False,**kwargs)
    elif l==m+1:
        plm = x*(2*m+1)*Associated_Legendre(m,m,x,is_array=False,**kwargs)
    else:
        pn2 = Associated_Legendre(m,m,x,is_array=False)
        pn1 = Associated_Legendre(m+1,m,x,is_array=False)
        for il in range(m+2,l+1):
            plm = ((2*il-1)*x*pn1-(il+m-1)*pn2)/(il-m)
            pn2,pn1 = pn1,plm

    if is_array:
        # print('is array')
        return plm.detach().cpu().numpy()
    else:
        return plm

def DAssociated_Legendre(l,m,x,**kwargs):
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    is_array = kwargs.pop('is_array',None)
    if is_array is None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device)
            is_array = True
        else:
            is_array = False
    P_l00 = kwargs.get('P_l00', Associated_Legendre(l  ,m,x,is_array=False,**kwargs))
    P_lp1 = kwargs.get('P_lp1', Associated_Legendre(l+1,m,x,is_array=False,**kwargs))
    ret   = -(l+1)*x*P_l00+(l-m+1)*P_lp1
    ret   = ret/(x**2-1)
    if is_array:
        ret = ret.detach().cpu().numpy()
    return ret

def Spherical_Harmonics(l,m,theta,phi, **kwargs):
    is_array = False
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    if isinstance(theta, np.ndarray) or isinstance(phi, np.ndarray):
        theta = torch.from_numpy(theta).to(device)
        phi   = torch.from_numpy(phi).to(device)
        is_array = True
    elif isinstance(theta, (int, float, complex)):
        theta = torch.tensor(theta).to(device)
        phi   = torch.tensor(phi  ).to(device)
    Plm = Associated_Legendre(l, m, torch.cos(theta),**kwargs)
    factorial1 = np.math.factorial(l-m)
    factorial2 = np.math.factorial(l+m)
    digit1     = len(str(factorial1))
    digit2     = len(str(factorial2))
    digit0     = 300
    if digit1<digit0 and digit2<digit0:
        Ylm = np.sqrt((2*l+1)/(4*np.pi)*float(factorial1)/float(factorial2))*Plm*torch.exp(1j*m*phi)
    elif digit1<digit0 and digit2>=digit0:
        Ylm = np.sqrt((2*l+1)/(4*np.pi)*float(factorial1)/float(factorial2/10**digit2))*Plm*torch.exp(1j*m*phi)
        for i in range(int(np.ceil(digit2/digit0))):
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
        Y_lm = Spherical_Harmonics(l,m,tt,pp,**kwargs)
        br_lm  = (Alm_lm * l * torch.pow(rr, l - 1) - (l + 1) * Blm_lm * torch.pow(rr, -l - 2))*Y_lm
        bp_lm  = (Alm_lm * torch.pow(rr, l-1) + Blm_lm * torch.pow(rr, -l - 2)) * 1j * m * Y_lm
        Y_lp1_m = Spherical_Harmonics(l+1,m,tt,pp,**kwargs)
        dY_dth  = 1/torch.sin(tt)*(-(l+1)*Y_lm*torch.cos(tt)+(l-m+1)*Y_lp1_m*np.sqrt((2*l+1)*(l+m+1)/(2*l+3)/(l-m+1)))
        bt_lm   = (Alm_lm*torch.pow(rr,l-1)+Blm_lm*torch.pow(rr,-l-2))*dY_dth
        # Y_l_mn1 = Spherical_Harmonics(l,m-1,tt,pp,**kwargs)
        # dY_dth  = 1/torch.sin(tt)*(-(l+m)*(l-m+1)*torch.abs(torch.sin(tt))*np.sqrt((2*l+1)/(2*l-1)*(l-m)/(l+m))*Y_l_mn1-m*torch.cos(tt)*Y_lm)

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
        T = np.arccos(Z/R)
        P = np.arctan(Y/X)
        P = np.where(X<0,P+np.pi,P)
    else:
        raise ValueError('Grid point `rtp` in spherical system or `xyz` in cartesian system should be provided...')
    bx = br*np.sin(T)*np.cos(P)+bt*np.cos(T)*np.cos(P)-bp*np.sin(P)
    by = br*np.sin(T)*np.sin(P)+bt*np.cos(T)*np.sin(P)+bp*np.cos(P)
    bz = br*np.cos(T)-bt*np.sin(T)
    b_vec = np.stack([bx,by,bz])
    return b_vec

def trilinear_interpolation(field, idx):
    xi,yi,zi    = np.floor(idx).astype(int).T if len(np.shape(idx))>1 else np.floor(idx).astype(int)
    xd,yd,zd    = (idx-np.floor(idx)).T
    nx,ny,nz    = field.shape[:3]
    xf,yf,zf    = np.where(xi+1>nx-1,nx-1,xi+1), np.where(yi+1>ny-1,ny-1,yi+1), np.where(zi+1>nz-1,nz-1,zi+1)
    f           = field
    out         = (xi>nx-1) | (yi>ny-1) | (zi>nz-1) | (xi<0) | (yi<0) | (zi<0)
    out         = out | (xf<0) | (yf<0) | (zf<0) | (xf>nx-1) | (yf>ny-1) | (zf>nz-1)
    xi,yi,zi    = np.where(xi>nx-1,nx-1,xi), np.where(yi>ny-1,ny-1,yi), np.where(zi>nz-1,nz-1,zi)
    xi,yi,zi    = np.where(xi<0,0,xi), np.where(yi<0,0,yi), np.where(zi<0,0,zi)
    xf,yf,zf    = np.where(xi+1>nx-1,nx-1,xi+1), np.where(yi+1>ny-1,ny-1,yi+1), np.where(zi+1>nz-1,nz-1,zi+1)
    xf,yf,zf    = np.where(xf<0,0,xf),np.where(yf<0,0,yf),np.where(zf<0,0,zf)
    if isinstance(xd, np.ndarray):
        xd              = xd[(slice(None),)+(np.newaxis,)*(f[xi,yi,zi].ndim-1)]
        yd              = yd[(slice(None),)+(np.newaxis,)*(f[xi,yi,zi].ndim-1)]
        zd              = zd[(slice(None),)+(np.newaxis,)*(f[xi,yi,zi].ndim-1)]
    ret         = f[xi,yi,zi]*(1-xd)*(1-yd)*(1-zd)+f[xf,yi,zi]*xd*(1-yd)*(1-zd)+f[xi,yf,zi]*(1-xd)*yd*(1-zd)+\
                  f[xi,yi,zf]*(1-xd)*(1-yd)*zd+f[xf,yf,zi]*xd*yd*(1-zd)+f[xf,yi,zf]*xd*(1-yd)*zd+\
                  f[xi,yf,zf]*(1-xd)*yd*zd+f[xf,yf,zf]*xd*yd*zd
    ret[out]=np.nan
    return ret

def Brtp_lm(l,m, rr, tt, pp,**kwargs):
    P_l00 = kwargs.get('P_l00', None)
    P_lp1 = kwargs.get('P_lp1', None)
    Alm   = kwargs.get('Alm'  , None)
    Blm   = kwargs.get('Blm'  , None)
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
        bp_lm  = (Alm_lm * torch.pow(rr, l-1) + Blm_lm * torch.pow(rr, -l - 2)) * 1j * m * Y_lm
        dY_dth  = 1/torch.sin(tt)*(-(l+1)*Y_lm*torch.cos(tt)+(l-m+1)*Y_lp1_m*np.sqrt((2*l+1)*(l+m+1)/(2*l+3)/(l-m+1)))
        bt_lm   = (Alm_lm*torch.pow(rr,l-1)+Blm_lm*torch.pow(rr,-l-2))*dY_dth

    if is_array:
        br_lm = br_lm.detach().cpu().numpy()
        bt_lm = bt_lm.detach().cpu().numpy()
        bp_lm = bp_lm.detach().cpu().numpy()
    return br_lm,bt_lm,bp_lm

def matplotlib_to_plotly(cmap, pl_entries=255):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
    return pl_colorscale


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
    B_r,B_p,B_t = Brtp
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

def data2vts(data_ls, name_ls, vts_name='data', **kwargs):
    """
    Convert a list of data arrays to a VTK structured grid file (.vts).

    Parameters:
    -----------
    data_ls : list of numpy.ndarray
        A list of data arrays to be saved. Each array should be either 3D or a tuple of 3D arrays.
    name_ls : list of str
        A list of names corresponding to each data array in `data_ls`.
    vts_name : str, optional
        The name of the VTK file to be saved (default is 'data').
    **kwargs : dict
        Additional keyword arguments:
        - xyz : tuple of numpy.ndarray
            Cartesian coordinates (x, y, z) for the structured grid.
        - rtp : tuple of numpy.ndarray
            Spherical coordinates (r, theta, phi) for the structured grid. If provided, they will be converted to Cartesian coordinates.

    Returns:
    --------
    None
        The function saves the data to a VTK file and does not return any value.

    Raises:
    -------
    TypeError
        If neither `xyz` nor `rtp` is provided.

    Notes:
    ------
    - The function uses `np.gradient` to measure the time taken for the conversion and saving process.
    - If the data array is not 3D, it is assumed to be a tuple of 3D arrays and is converted accordingly.
    - The VTK file is saved using the `gridToVTK` function from the `evtk.hl` module.

    Example:
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 10)
    >>> y = np.linspace(0, 1, 10)
    >>> z = np.linspace(0, 1, 10)
    >>> X, Y, Z = np.meshgrid(x, y, z)
    >>> data1 = np.sin(X) * np.cos(Y) * np.cos(Z)
    >>> data2 = np.cos(X) * np.sin(Y) * np.sin(Z)
    >>> data2vts([data1, data2], ['data1', 'data2'], vts_name='example', xyz=(X, Y, Z))
    Save data to example.vts.
    Time Used: 0.001 min...
    """
    t0 = time.time()
    xyz = kwargs.get('xyz', None)
    rtp = kwargs.get('rtp', None)
    if xyz is None and rtp is None:
        return TypeError('`xyz` or `rtp` should be provided at least one...')
    X,Y,Z = xyz if xyz is not None else rtp2xyz(rtp)
    pointData = dict()
    for name,data in zip(name_ls,data_ls):
        pointData[name]=data if data.ndim==3 else tuple(idata for idata in data)
    gridToVTK(vts_name, X, Y, Z, 
              pointData=pointData)
    print(f'Save data to {vts_name}.vts.\nTime Used: {(time.time()-t0)/60:8.3} min...')