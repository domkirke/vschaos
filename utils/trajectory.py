import numpy as np, torch, scipy.ndimage as ndimage
from . import checklist


def line(z_dim, n_steps):
    origins = np.random.multivariate_normal(np.zeros((z_dim)), np.diag(3*np.ones((z_dim))), 2)
    coord_interp = np.linspace(0, 1, n_steps)
    z_interp = np.zeros((len(coord_interp), origins.shape[1]))
    for i,y in enumerate(coord_interp):
        z_interp[i] = ndimage.map_coordinates(origins, [y * np.ones(origins.shape[1]), np.arange(origins.shape[1])], order=2)
    z_traj = torch.from_numpy(z_interp)
    return z_traj


def get_random_trajectory(trajectory_type, z_dim, n=1, n_steps=1000, **kwargs):
    trajectories = []
    trajectory_type = checklist(trajectory_type)
    for traj_type in trajectory_type:
        if traj_type in trajectory_hash.keys():
            for i in range(n):
                trajectories.append(trajectory_hash[traj_type](z_dim, n_steps, **kwargs))
        else:
            raise LookupError('trajectory type %s not known'%traj_type)
    return trajectories


def get_interpolation(origins, n_steps=1000, interp_order=2, **kwargs):
    if len(origins.shape) > 2:
        # is a sequence ; get homotopies
        return torch.stack([get_interpolation(origins[:,i], n_steps, interp_order=interp_order, **kwargs) for i in range(origins.shape[1])], dim=1)
    device = torch.device('cpu')
    if torch.is_tensor(origins):
        device = origins.device
        origins = origins.cpu().detach().numpy()
    coord_interp = np.linspace(0, 1, n_steps)
    z_interp = np.zeros((len(coord_interp), origins.shape[1]))
    for i,y in enumerate(coord_interp):
        z_interp[i] = ndimage.map_coordinates(origins, [y * np.ones(origins.shape[1]), np.arange(origins.shape[1])], order=interp_order)
    z_traj = torch.from_numpy(z_interp).to(device=device)
    return z_traj


trajectory_hash = {'line':line}
