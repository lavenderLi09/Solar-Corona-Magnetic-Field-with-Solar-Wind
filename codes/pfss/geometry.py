from .needs import *

def Generalized_Jacobian(Vfield, Lame, dq=None, **kwargs):
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    if isinstance(Vfield, np.ndarray):
        is_array = True
        Vfield   = torch.from_numpy(Vfield).to(device)
        Lame     = torch.from_numpy(Lame  ).to(device)
    else:
        is_array = False
    d1,d2,d3 = dq if dq is not None else [1,1,1]
    H1,H2,H3 = Lame
    H        = H1*H2*H3
    V1,V2,V3 = Vfield
    F1,F2,F3 = Lame*Vfield
    G1       = V1*H2*H3
    G2       = H1*V2*H3
    G3       = H1*H2*V3
    GF1      = torch.stack([G1,F2,F3])
    GF2      = torch.stack([F1,G2,F3])
    GF3      = torch.stack([F1,F2,G3])
    pd1      = torch.cat([(GF1[:,1:2,:,:]*4-GF1[:,0:1,:,:]*3  -GF1[:,2:3,:,:])/2,
                          (GF1[:,2: ,:,:]  -GF1[:,:-2,:,:])/2,
                          (GF1[:,-1:,:,:]*3-GF1[:,-2:-1,:,:]*4-GF1[:,-3:-2,:,:])/2
                         ], dim=1)/d1
    pd2      = torch.cat([(GF2[:,:,1:2,:]*4-GF2[:,:,0:1  ,:]*3-GF2[:,:, 2: 3,:])/2,
                          (GF2[:,:,2: ,:]  -GF2[:,:,:-2  ,:])/2,
                          (GF2[:,:,-1:,:]*3-GF2[:,:,-2:-1,:]*4-GF2[:,:,-3:-2,:])/2
                         ], dim=2)/d2
    pd3      = torch.cat([(GF3[:,:,:,1:2]*4-GF3[:,:,:,0:1  ]*3-GF3[:,:,:, 2: 3])/2,
                          (GF3[:,:,:,2: ]  -GF3[:,:,:,:-2  ])/2,
                          (GF3[:,:,:,-1:]*3-GF3[:,:,:,-2:-1]*4-GF3[:,:,:,-3:-2])/2
                         ], dim=3)/d3
    GA       = torch.stack([pd1,pd2,pd3], dim=0)/H
    if is_array:
        return GA.detach().cpu().numpy()
    else:
        return GA

def Lame_coefficient(geometry='spherical', **kwargs):
    if geometry=='spherical':
        rtp = kwargs.get('rtp', None)
        if rtp is None:
            raise ValueError('Please give a spherical coordinates `rtp`...')
        r,t,p = rtp
        dr    = r[1,0,0]-r[0,0,0]
        dt    = t[0,1,0]-t[0,0,0]
        dp    = p[0,0,1]-p[0,0,0]
        dq    = [dr,dt,dp]
        H1    = np.ones_like(r)
        H2    = r
        H3    = r*np.sin(t)
        Lame  = np.stack([H1,H2,H3])
    elif geometry=='cylindrical':
        rpz = kwargs.get('rpz', None)
        if rpz is None:
            raise ValueError('Please give a cylindrical coordinates `rpz`...')
        r,p,z = rpz
        dr    = r[1,0,0]-r[0,0,0]
        dp    = p[0,1,0]-p[0,0,0]
        dz    = z[0,0,1]-z[0,0,0]
        dq    = [dr,dp,dz]
        H1    = np.ones_like(r)
        H2    = np.ones_like(r)
        H3    = r
        Lame  = np.stack([H1,H2,H3])
    elif geometry=='cartesian':
        xyz  = kwargs.get('xyz' , None)
        dxyz = kwargs.get('dxyz', None)
        if dxyz is None:
            if xyz is None:
                raise ValueError('Please give a cartesian coordinates `xyz`...')
            else:
                x,y,z = xyz
                dx = x[1,0,0]-x[0,0,0]
                dy = y[0,1,0]-y[0,0,0]
                dz = z[0,0,1]-z[0,0,0]
                dq = [dx,dy,dz]
                Lame = np.zeros(xyz.shape)
    else:
        raise ValueError("`geometry` only support to be 'spherical','cylindrical' and 'cartesian'...")
    return Lame, dq

def div(V, geometry='spherical', **kwargs):
    Lame, dq = Lame_coefficient(geometry, **kwargs)
    GA       = Generalized_Jacobian(V, Lame, dq=dq, **kwargs)
    ret      = GA[0,0]+GA[1,1]+GA[2,2]
    return ret

def rot(V, geometry='spherical', **kwargs):
    Lame, dq = Lame_coefficient(geometry, **kwargs)
    GA       = Generalized_Jacobian(V, Lame, dq=dq, **kwargs)
    ret      = np.stack([(GA[1,2]-GA[2,1])*Lame[0],
                         (GA[2,0]-GA[0,2])*Lame[1],
                         (GA[0,1]-GA[1,0])*Lame[2]
                        ], axis=0)
    return ret

def grad(V, geometry='spherical', **kwargs):
    Lame, dq = Lame_coefficient(geometry, **kwargs)
    h1,h2,h3 = Lame
    g1,g2,g3 = np.gradient(V,axis=(0,1,2))
    ret      = np.stack([g1/(h1*dq[0]),g2/(h2*dq[1]),g3/(h3*dq[2])],axis=0)
    return ret

def dot(V1,V2, **kwargs):
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    if isinstance(V1, np.ndarray):
        V1 = torch.from_numpy(V1).to(device)
        V2 = torch.from_numpy(V2).to(device)
        is_array = True
    else:
        is_array = False
    ret = torch.sum(V1*V2, dim=0)
    if is_array:
        return ret.detach().cpu().numpy()
    else:
        return ret

def cross(V1,V2, **kwargs):
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    if isinstance(V1, np.ndarray):
        V1 = torch.from_numpy(V1).to(device)
        V2 = torch.from_numpy(V2).to(device)
        is_array = True
    else:
        is_array = False
    ret = torch.cross(V1,V2, dim=0)
    if is_array:
        return ret.detach().cpu().numpy()
    else:
        return ret

def xyz2rtp(xyz):
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).
    
    Parameters:
    x (float): x-coordinate
    y (float): y-coordinate
    z (float): z-coordinate
    
    Returns:
    tuple: (r, theta, phi)
        r (float): radial distance
        theta (float): polar angle (in radians)
        phi (float): azimuthal angle (in radians)
    """
    x,y,z=xyz
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.where(r!=0,np.arccos(z/r),0)
    phi = np.arctan2(y, x) % (np.pi*2)
    return np.stack([r, theta, phi])

def rtp2xyz(rtp):
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z).
    
    Parameters:
    r (float): radial distance
    theta (float): polar angle (in radians)
    phi (float): azimuthal angle (in radians)
    
    Returns:
    tuple: (x, y, z)
        x (float): x-coordinate
        y (float): y-coordinate
        z (float): z-coordinate
    """
    r,theta,phi=rtp
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z])

def Vrtp2Vxyz(Vrtp, rtp):
    """
    Convert a vector from spherical coordinates to Cartesian coordinates.

    Parameters:
    Vrtp: A list or array of three elements, representing the vector in spherical coordinates (Vr, Vtheta, Vphi).
    rtp: A list or array of three elements, representing the position in spherical coordinates (r, theta, phi).
         Theta and phi should be in radians.

    Returns:
    Vxyz: An array of three elements, representing the vector in Cartesian coordinates (Vx, Vy, Vz).
    """
    Vr,Vt,Vp = Vrtp
    r , t, p = rtp
    # Calculate the direction of radial unit vector
    e_rx = np.sin(t)*np.cos(p)
    e_ry = np.sin(t)*np.sin(p)
    e_rz = np.cos(t)
    # Calculate the direction of the polor angle unit vector (from z-axis to xy-plane)
    e_tx = np.cos(t)*np.cos(p)
    e_ty = np.cos(t)*np.sin(p)
    e_tz =-np.sin(t)
    # Calculate the direction of the azimuthal angle unit vector (in xy-plane)
    e_px =-np.sin(p)
    e_py = np.cos(p)
    e_pz = 0
    # Convert the vector from spherical to Cartesian coordinates
    Vx = Vr*e_rx+Vt*e_tx+Vp*e_px
    Vy = Vr*e_ry+Vt*e_ty+Vp*e_py
    Vz = Vr*e_rz+Vt*e_tz+Vp*e_pz
    Vxyz = np.stack([Vx,Vy,Vz])
    return Vxyz